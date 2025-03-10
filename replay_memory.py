import itertools
from collections import defaultdict
import numpy as np
import json
from gym.spaces import Space
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from util import discounted_seq_max

class SequentialDataset(Dataset):
    def __init__(self, file_path: str):
        self.trj_dict = defaultdict(list)
        self.sample_keys = ['observations', 'actions', 'guides']
        self.state_scaler, self.guide_scaler = StandardScaler(), StandardScaler()

        # Load dataset and split into the trajectories
        dataset_dict = np.load(file_path, allow_pickle=True).item()
        ep_end_indices = np.where(dataset_dict['end'] == 1)[0]
        start_idx = 0
        for end_idx in ep_end_indices:
            for k in ['observations', 'actions']:
                self.trj_dict[k].append(dataset_dict[k][start_idx:end_idx+1])
            # Defualt guide function: Return-To-Go
            self.trj_dict['guides'].append(
                np.flip(np.cumsum(np.flip(dataset_dict['rewards'][start_idx:end_idx+1], axis=0)), axis=0).reshape(-1,1)
            )
            start_idx = end_idx + 1

        # Train scalers
        self.state_scaler.fit(dataset_dict['observations'])
        self.guide_scaler.fit(np.fromiter(itertools.chain(*self.trj_dict['guides']), dtype=float).reshape(-1, 1))

    def __len__(self):
        return len(self.trj_dict['observations'])
    
    def __getitem__(self, idx):
        sampled_dict = {k: self.trj_dict[k][idx] for k in self.sample_keys}
        #sampled_dict['observations'] = self.state_scaler.transform(sampled_dict['observations'])
        sampled_dict['guides'] = self.guide_scaler.transform(sampled_dict['guides'])
        return sampled_dict
    
    def set_guide(self, calc_guide_function) -> None:
        """
        Replace guide with the new value calculated by passed function
        """
        g = calc_guide_function(self.trj_dict['observations'])
        self.guide_scaler.fit(np.fromiter(itertools.chain(*g), dtype=float).reshape(-1, 1))
        self.trj_dict['guides'] = discounted_seq_max(g)

    def set_critic_guide(self, critic_path: str, obs_dim: int, action_space: Space) -> None:
        """
        Replace guide with the new value calculated by ciritic network
        (discounted expected sequential maximum state-value)
        """
        from distributional.codac import CODAC
        
        with open(critic_path+'.json', 'r', encoding='UTF-8') as f:
            critic_param_dict = json.load(f)
        critic_model = CODAC(
            obs_dim,
            action_space,
            **critic_param_dict
        )
        critic_model.load_model(critic_path+'_weight') 
        g = critic_model.calc_critics(self.trj_dict['observations'])
        self.guide_scaler.fit(np.fromiter(itertools.chain(*g), dtype=float).reshape(-1, 1))
        self.trj_dict['guides'] = discounted_seq_max(g)

    def get_all_obs(self):
        return self.trj_dict['observations']

    def state_scaling(self, state: np.ndarray) -> np.ndarray:
        return self.state_scaler.transform(state)
    
    def guide_scaling(self, guide: np.ndarray) -> np.ndarray:
        return self.guide_scaler.transform(guide)

    
    