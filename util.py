import sys
import json
import argparse
from typing import Any, Tuple

from collections import defaultdict
import numpy as np
import importlib
import gym


def str_to_type(str_type: str) -> type:
    type_mapping = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool
    }
    return type_mapping.get(str_type, Any)

def load_params(auto_tuning: bool=True, rl_algorithm: str='DT', env_type: str= 'pointmass') -> Tuple[dict, dict]:
    """
    Load parameters from config.json and command-line arguments.

    Args:
        auto_tuning (bool): Whether to use hyperparameter auto-tuning or not.
        rl_algorithm (str): Selected RL algorithm.
        env_type (str): Selected environment.

    Returns:
        Tuple[dict, dict]: 
            - Dictionary of fixed parameters.
            - Dictionary of tunable parameters.
    """
    # Set RL algorithm
    if not str(sys.argv[1]).startswith('--'):
        rl_algorithm = sys.argv.pop(1)
    if not str(sys.argv[1]).startswith('--'):
        env_type = sys.argv.pop(1)

    # Load selected algorithm's default settings
    with open('config.json', 'r', encoding='UTF-8') as f:
        param_dicts = json.load(f)
        default_params = param_dicts[rl_algorithm]
        env_params = param_dicts[env_type]

    # Create dicts
    parser = argparse.ArgumentParser()
    tunable_param_dict = {}
    for param, attrs in default_params.items():
        if auto_tuning and (t := attrs.get('tuning')):
            tunable_param_dict[param] = {
                'tuning': t,
                'range': attrs.get('range'),
                'step': attrs.get('step')
            }
        else:
            parser.add_argument(
                '--' + param, 
                type = str_to_type(attrs.get('type')), 
                default = attrs.get('default')
            )
    fixed_param_dict = vars(parser.parse_args())
    fixed_param_dict['rl_algorithm'] = rl_algorithm
    fixed_param_dict['env_settings'] = env_params

    return fixed_param_dict, tunable_param_dict

def create_env(env_properties: dict) -> gym.Env:
    """
    Initialize environment class(using dynamic import)
    """
    env_module = importlib.import_module(env_properties.pop('module_path'))
    env_cls = getattr(env_module, env_properties.pop('class_name'))
    return env_cls(**env_properties)

def pad_seq(seq: np.ndarray, cur_seq_len: int, max_seq_len: int) -> np.ndarray:
    """
    Padding the passed sequence array with zero-value
    """
    seq = seq.reshape(cur_seq_len, -1)
    return np.concatenate([np.zeros((max_seq_len-cur_seq_len, seq.shape[1])), seq], axis=0)

def batch_collate(batch: list, max_seq_len: int=100) -> defaultdict:
    """
    Function used in DataLoader as collate_fn
    """
    collated_batch = defaultdict(list)
    for sampled_seq_dict in batch:
        # key, seq_array
        seq_len = len(sampled_seq_dict['observations'])
        collated_batch['timesteps'].append(pad_seq(np.arange(seq_len)+1, seq_len, max_seq_len))
        collated_batch['masks'].append(pad_seq(np.ones(seq_len), seq_len, max_seq_len))
        for k, v in sampled_seq_dict.items():
            collated_batch[k].append(pad_seq(v, seq_len, max_seq_len))
    
    # collated_batch[key][n_seq]
    return {k: np.array(v) for k, v in collated_batch.items()}

def discounted_seq_max(seq_values: list, gamma: float=0.99) -> list:
    guide_list = []
    for seq in seq_values:
        guide = np.zeros_like(seq)
        start_idx = 0
        while len(seq) > 0 :
            expeted_max_idx = np.argmax(seq)
            for t in range(expeted_max_idx + 1) :
                guide[start_idx + t] = seq[expeted_max_idx] * ((gamma)**(expeted_max_idx - t))
            seq = seq[(expeted_max_idx + 1):]
            start_idx += (expeted_max_idx + 1)
        guide_list.append(guide)

    return guide_list
