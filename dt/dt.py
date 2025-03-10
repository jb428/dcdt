from dt.dt_algorithms import BaseDT, DCDT, OriginalDT
import torch
import time
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_

def algo_type_mapping(algo_type: str) -> BaseDT:
    algo_type_dict = {
        'DT': OriginalDT,
        'DCDT': DCDT,
    }  
    return algo_type_dict[algo_type]

class DT:
    def __init__(
        self,
        state_dim,
        act_dim,
        loss_fn,
        param_dict
    ):
        # Default values
        self.rl_algorithm='DT'
        self.lr = 0.0001
        self.weight_decay = 1e-4
        self.device = 'cuda'
        self.dropout = 0.1
        # Update by param_dict
        self.__dict__.update(param_dict)

        dt_cls = algo_type_mapping(self.rl_algorithm)
        self.nn_transformer = dt_cls(
            state_dim=state_dim,
            act_dim=act_dim,
            resid_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            **param_dict
        )
        self.nn_transformer.eval()

        self.loss_fn = loss_fn
        self.optimizer = torch.optim.AdamW(
            self.nn_transformer.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

    def update_parameters(self, batch_data) :
        self.nn_transformer.train()

        states = torch.from_numpy(batch_data['observations']).to(dtype=torch.float32, device=self.device)
        actions = torch.from_numpy(batch_data['actions']).to(dtype=torch.float32, device=self.device)
        guides = torch.from_numpy(batch_data['guides']).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(batch_data['masks']).to(dtype=torch.long, device=self.device).reshape(states.shape[0], -1)
        timesteps = torch.from_numpy(batch_data['timesteps']).to(dtype=torch.long, device=self.device).reshape(states.shape[0], -1)

        action_target = torch.clone(actions)
        guide_target = torch.clone(guides)

        _, action_preds, guide_preds = self.nn_transformer.forward(
            states, actions, guides, timesteps, attention_mask=masks,
        )
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[masks.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[masks.reshape(-1) > 0]

        guide_preds = guide_preds.reshape(-1, 1)[masks.reshape(-1) > 0]
        guide_target = guide_target.reshape(-1, 1)[masks.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, guide_preds,
            None, action_target, guide_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.nn_transformer.parameters(), .25)
        self.optimizer.step()

        a_loss = torch.mean((action_preds-action_target)**2).detach().cpu().item()
        g_loss = torch.mean((guide_preds-guide_target)**2).detach().cpu().item()

        self.nn_transformer.eval()

        return loss.detach().cpu().item(), a_loss, g_loss

    def get_action(self, states, actions, guides, timesteps) : 
        return self.nn_transformer.pred_action(states, actions, guides, timesteps)
    
    def get_pred(self, **kwargs):
        """
        Args:
            Pass padded np.ndarray of states, actions, guides, timesteps, masks.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - action_pred 
                - guide_pred 
        """
        return self.nn_transformer.pred(**kwargs)

    
    def save_model(self, path) :
        torch.save(self.nn_transformer.state_dict(), path + '.pt')

    def load_model(self, path) :
        self.nn_transformer.load_state_dict(torch.load(path, map_location=self.device))