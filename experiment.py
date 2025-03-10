import functools
import copy
import mlflow
import mlflow.experiments
import optuna
import gym
import importlib

from util import create_env, load_params

def suggest_param(trial: optuna.trial.Trial, param_name: str, param_attrs: dict):
    """
    Set tunable parameters 
    """
    suggest_dict = {
        'int': trial.suggest_int,
        'float': trial.suggest_float,
        'categorical': trial.suggest_categorical
    }

    if param_attrs['tuning'] == 'categorical':
        return suggest_dict[param_attrs['tuning']](
            name=param_name, 
            choices=param_attrs['range']
        )
    elif 'step' in param_attrs.keys():
        return suggest_dict[param_attrs['tuning']](
            name=param_name,
            low=param_attrs['range'][0],
            high=param_attrs['range'][1],
            step=param_attrs['step']
        )
    else:
        return suggest_dict[param_attrs['tuning']](
            name=param_name,
            low=param_attrs['range'][0],
            high=param_attrs['range'][1]
        )

def objective(trial: optuna.trial.Trial, fixed_params: dict, tunable_params: dict):
    """
    Searching hyperparameters
    """
    params_dict = copy.deepcopy(fixed_params)
    for param, attrs in tunable_params.items():
        params_dict[param] = suggest_param(trial, param, attrs)

    env = create_env(params_dict['env_settings'])
    params_dict['max_ep_len'] = env.max_timestep
    print(params_dict)
    agent_cls = getattr(importlib.import_module('agent'), fixed_params['rl_algorithm']+'Agent')
    agent = agent_cls(env, dataset_path=None, param_dict=params_dict)
    
    score_hist = []
    with mlflow.start_run(
        run_name=fixed_params['rl_algorithm']+'_'+
            fixed_params['env_settings']['class_name']+'_'+
            str(trial.number),
        tags={
            'dt_type': fixed_params['rl_algorithm'],
            'env_type': fixed_params['env_settings']['class_name'],
        },
        description='Searching hyperparameters'
    ):
        mlflow.log_params({k: v for k, v in params_dict.items() if k in tunable_params.keys()})
        # Train model(agent.train)
        for epoch in range(fixed_params['epoch']):
            mlflow.log_metrics(agent.train())
            # Get score(agent.evaluate)
            if (epoch+1) % fixed_params['eval_interval'] == 0:
                score_mean, score_std = agent.evaluate()
                mlflow.log_metrics({'score_mean': score_mean, 'score_std': score_std})
                score_hist.append(score_mean)
            # Save the best model(using agent method)
            if (epoch+1) % fixed_params['save_interval'] == 0:
                mlflow.pytorch.log_model(agent.export_model(), artifact_path="model_"+str(epoch+1))
        
        mlflow.pytorch.log_model(agent.export_model(), artifact_path="model_final")
                
    return max(score_hist)

def experiment(fixed_params: dict, tunable_params: dict) -> None:
    """
    Create optuna study
    """
    mlflow.set_experiment('DT-based')

    # Warpping objective function to pass parameters
    wrapped_objective = functools.partial(objective, fixed_params=fixed_params, tunable_params=tunable_params)

    # Define experiment with Optuna
    # To search the best hyperparameters
    study = optuna.create_study(
        direction=fixed_params.get('optuna_direction', 'maximize')
        #,pruner=optuna.pruners.MedianPruner(n_warmup_steps=fixed_params.get('optuna_warmup_steps', 1000))
    )
    study.optimize(wrapped_objective, n_trials=fixed_params.get('optuna_trials', 10))

    with mlflow.start_run():
        mlflow.log_artifact('config.json', artifact_path='configs')

if __name__=='__main__':
    experiment(*load_params(auto_tuning=True))


