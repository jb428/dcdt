from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import transformers

from dt.trajectory_gpt2 import GPT2Model

class BaseDT(nn.Module, ABC):
    """
    Abstract class of decision transformer based algorithms
    """
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

    @abstractmethod
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        raise NotImplementedError
    
    @abstractmethod
    def pred(self, states, actions, guides, timesteps, masks, **kwargs):
        # these will come as tensors on the correct device
        raise NotImplementedError

class DCDT(BaseDT):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            n_head,
            hidden_size,
            max_ep_len,
            device,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim)
        self.hidden_size = hidden_size
        self.device = device
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_head=n_head,
            n_embd=hidden_size,
            n_ctx = 512,
            **kwargs
        )
        
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config).to(self.device)

        self.embed_timestep = nn.Embedding(max_ep_len+1, hidden_size).to(self.device)
        self.embed_return = torch.nn.Linear(1, hidden_size).to(self.device)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size).to(self.device)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size).to(self.device)

        self.embed_ln = nn.LayerNorm(hidden_size).to(self.device)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim).to(self.device)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        ).to(self.device)
        self.predict_return = torch.nn.Linear(hidden_size, 1).to(self.device).to(self.device)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        # modified : (s_1, g_1, a_1, s_2, g_2, a_2, ...)
        stacked_inputs = torch.stack(
            (state_embeddings, returns_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

        # modified : states (0), guide (1), actions (2)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        return_preds = self.predict_return(x[:,0])  # predict next return given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def pred(self, states, actions, guides, timesteps, masks, **kwargs):
        s = torch.from_numpy(states).to(device=self.device, dtype=torch.float32).reshape(1, -1, self.state_dim)
        a = torch.from_numpy(actions).to(device=self.device, dtype=torch.float32).reshape(1, -1, self.act_dim)
        g = torch.from_numpy(guides).to(device=self.device, dtype=torch.float32).reshape(1, -1, 1)
        t = torch.from_numpy(timesteps).to(device=self.device, dtype=torch.long).reshape(1, -1)
        m = torch.from_numpy(masks).to(device=self.device, dtype=torch.long).reshape(1, -1)

        _, action_preds, guide_preds = self.forward(s, a, g, t, attention_mask=m, **kwargs)

        return action_preds[0,-1].detach().cpu().numpy(), guide_preds[0,-1].detach().cpu().numpy()


class OriginalDT(BaseDT):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            n_head,
            hidden_size,
            max_ep_len,
            device,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim)
        self.hidden_size = hidden_size
        self.device = device
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_head=n_head,
            n_embd=hidden_size,
            n_ctx = 512,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config).to(self.device)

        self.embed_timestep = nn.Embedding(max_ep_len+1, hidden_size).to(self.device)
        self.embed_return = torch.nn.Linear(1, hidden_size).to(self.device)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size).to(self.device)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size).to(self.device)

        self.embed_ln = nn.LayerNorm(hidden_size).to(self.device)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim).to(self.device)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        ).to(self.device)
        self.predict_return = torch.nn.Linear(hidden_size, 1).to(self.device).to(self.device)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds
    
    def pred(self, states, actions, guides, timesteps, masks, **kwargs):
        s = torch.from_numpy(states).to(device=self.device, dtype=torch.float32).reshape(1, -1, self.state_dim)
        a = torch.from_numpy(actions).to(device=self.device, dtype=torch.float32).reshape(1, -1, self.act_dim)
        g = torch.from_numpy(guides).to(device=self.device, dtype=torch.float32).reshape(1, -1, 1)
        t = torch.from_numpy(timesteps).to(device=self.device, dtype=torch.long).reshape(1, -1)
        m = torch.from_numpy(masks).to(device=self.device, dtype=torch.long).reshape(1, -1)

        _, action_preds, _ = self.forward(s, a, g, t, attention_mask=m, **kwargs)

        return action_preds[0,-1].detach().cpu().numpy(), None
