# DCDT

 DCDT(Distributional Critic guided Decision Transformer) is a risk-averse offline reinforcement learning method based on the Decision Transformer, using distributional reinforcement learning agent as the risk estimator.  

<br/>

## Overview

 DCDT uses risk-aware state values as the guide value instead of return-to-go. The state-values are obtained from CODAC critic network and undergo CVaR evaluation. Then, the guide values, which represent the discounted maximum state values, are sequentially calculated. By using these values, DCDT can avoid overly optimistic predictions and overcome the limitations of the Decision Transformer in stochastic environments.

<br/>

### Training process of DCDT

<br/>

 ![train_dcdt](https://github.com/user-attachments/assets/9ee9de46-e3f2-48dc-8de7-f59e1d949a34)

<br/>

### Calculation of guide values
 ![calc_guide](https://github.com/user-attachments/assets/8ebfb2a7-aa6b-4fd9-8c07-3946b681d99e)

<br/><br/><br/>

## Get started

### Requirements
 This repository requires Python (>3.8), Pytorch (>2.0), Transformer (<=4.5.1). To train DCDT, a dataset and the pre-trained weights of CODAC are required. You can use the included ```Dockerfile``` for execution(Mujoco is not included, it must be installed separately for D4RL tests). 

<br/>
 
### Train model
 You can run an MLflow experiment with Optuna hyperparameter tuning as specified in ```config.json``` by executing the following command.

```
python experiment.py DCDT <environment> [options]
```

<br/>

## Contact
jaebbok@gmail.com

<br/>

## Acknowledgments
This project incorporates code and concepts from the following works:

* **Decision Transformer** : Our implementation is based on the official Decision Transformer codebase ([GitHub link](https://github.com/kzl/decision-transformer)) and the paper : \
Chen, Lili, et al. "Decision transformer: Reinforcement learning via sequence modeling." Advances in neural information processing systems 34 (2021): 15084-15097.

* **CODAC** : We utilize parts of the implementation from the official CODAC repository ([GitHub link](https://github.com/JasonMa2016/CODAC)) and refer to the original paper : \
Ma, Yecheng, Dinesh Jayaraman, and Osbert Bastani. "Conservative offline distributional reinforcement learning." Advances in neural information processing systems 34 (2021): 19235-19247.

We sincerely appreciate the authors' contributions to the research community.
