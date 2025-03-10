import time
import json
import importlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod
import gym
import torch
from torch.utils.data import Dataset, DataLoader

from replay_memory import SequentialDataset
from util import batch_collate, pad_seq

class BaseOfflineAgent(ABC):
    def __init__(self, eval_env, dataset_path, param_dict):
        self.model = None
        self.batch_size = None
        self.num_train_repeat = 1
        self.num_eval_repeat = 20
        self.eval_env = eval_env
        self.state_dim = eval_env.observation_space.shape[0]
        self.action_dim = eval_env.action_space.shape[0]
        self.max_timestep = eval_env.max_timestep
        self.dataset_path = dataset_path
        self.rtg = 0.
        self.__dict__.update(param_dict)
        
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path):
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self, path):
        raise NotImplementedError
    
class DTAgent(BaseOfflineAgent):
    def __init__(self, eval_env, dataset_path, param_dict):
        super().__init__(eval_env, dataset_path, param_dict)

        np.random.seed(param_dict['seed'])
        torch.manual_seed(param_dict['seed'])
        torch.cuda.manual_seed_all(param_dict['seed'])
        eval_env.seed(param_dict['seed'])

        self.memory = SequentialDataset(file_path=self.dataset_path+'.npy')

        from dt.dt import DT
        self.model = DT(
            eval_env.observation_space.shape[0],
            eval_env.action_space.shape[0],
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            param_dict=param_dict
        )
        
    def train(self):
        trj_loader = DataLoader(
            self.memory, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=batch_collate
        )
        loss_list = []
        for train_steps, sample in enumerate(trj_loader):
            if train_steps >= self.num_train_repeat:
                break
            loss_list.append(self.model.update_parameters(sample))

        mean_loss = np.mean(loss_list, axis=0) #[loss, a_loss, g_loss]
        return {'loss': mean_loss[0]}
    
    def evaluate(self, set_rtg=None):
        scores = np.zeros(self.num_eval_repeat, dtype=np.float32)
        for ep in range(self.num_eval_repeat):
            # keys = ['states', 'actions', 'guides', 'timesteps', 'masks']
            rtg = set_rtg if set_rtg is not None else self.rtg
            hist = defaultdict(list)
            hist['guides'].append(rtg) 
            hist['states'].append(self.eval_env.reset())
            for t in range(self.max_timestep):
                hist['timesteps'].append(t)
                hist['masks'].append(1.)
                hist['actions'].append(np.zeros(self.action_dim)) # dummy value
                _, cur_action = self.model.get_pred(**{k: pad_seq(np.asarray(v), t+1, self.max_timestep) for k, v in hist.items()})
                hist['actions'][-1] = cur_action

                # Env step
                next_state, reward, done, _ = self.eval_env.step(cur_action)
                hist['guides'].append(hist['guides'][-1] - reward)
                hist['states'].append(next_state)
                scores[ep] += reward
                if done:
                    break

        return np.mean(scores), np.std(scores)
    
    def save_model(self, path):
        self.model.save_model(path)
    
    def load_model(self, path):
        self.model.load_model(path)

    def export_model(self):
        return self.model.nn_transformer
    
class DCDTAgent(BaseOfflineAgent):
    def __init__(self, eval_env, dataset_path, param_dict):
        super().__init__(eval_env, dataset_path, param_dict)
        
        np.random.seed(param_dict['seed'])
        torch.manual_seed(param_dict['seed'])
        torch.cuda.manual_seed_all(param_dict['seed'])
        eval_env.seed(param_dict['seed'])

        self.memory = SequentialDataset(file_path=self.dataset_path+'.npy')
        self.memory.set_critic_guide(
            self.dataset_path, 
            self.eval_env.observation_space.shape[0], 
            self.eval_env.action_space
        )

        from dt.dt import DT
        self.model = DT(
            eval_env.observation_space.shape[0],
            eval_env.action_space.shape[0],
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: \
                torch.mean((a_hat - a)**2) + torch.mean((r_hat - r)**2),
            param_dict=param_dict
        )
    
    def train(self):
        trj_loader = DataLoader(
            self.memory, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=batch_collate
        )
        loss_list = []
        for train_steps, sample in enumerate(trj_loader):
            if train_steps >= self.num_train_repeat:
                break
            loss_list.append(self.model.update_parameters(sample))
        mean_loss = np.mean(loss_list, axis=0) #[loss, a_loss, g_loss]
        return {'loss': mean_loss[0], 'action_loss': mean_loss[1], 'guide_loss': mean_loss[2]}
    
    def evaluate(self, rtg=None):
        scores = np.zeros(self.num_eval_repeat, dtype=np.float32)
        for ep in range(self.num_eval_repeat):
            # keys = ['states', 'actions', 'guides', 'timesteps', 'masks']
            hist = defaultdict(list) 
            hist['states'].append(self.eval_env.reset())
            for t in range(self.max_timestep):
                hist['timesteps'].append(t)
                hist['masks'].append(1.)
                hist['guides'].append(0.) # dummy value
                hist['actions'].append(np.zeros((1, self.action_dim))) # dummy value
                _, cur_guide = self.model.get_pred(**{k: pad_seq(np.asarray(v), t+1, self.max_timestep) for k, v in hist.items()})
                hist['guides'][-1] = cur_guide[0]
                cur_action, _  = self.model.get_pred(**{k: pad_seq(np.asarray(v), t+1, self.max_timestep) for k, v in hist.items()})
                hist['actions'][-1] = np.array(cur_action).reshape(1, self.action_dim)

                # Env step
                next_state, reward, done, _ = self.eval_env.step(cur_action)
                hist['states'].append(next_state)
                scores[ep] += reward
                if done:
                    break
        return np.mean(scores), np.std(scores)
            
    def save_model(self, path):
        self.model.save_model(path)
    
    def load_model(self, path):
        self.model.load_model(path)

    def export_model(self):
        return self.model.nn_transformer

