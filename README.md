# We conduct experiments on two environments, Two Color Grid World and mujoco-ant, the running command line for these two environments are not similar.
# Two Color Grid World

If you want to conduct experiment on Two Color Grid World, here's the command line:
cd h_divergence_meta_learning/bmg
python sac_bmg.py
You can check out Two Color Grid World environment in h_divergence_meta_learning/bmg/tcgw_env.py, which defines the environment.
You can also check out some parameters of the experiment in h_divergence_meta_learning/bmg/sac_bmg.py, below are some demonstration of the parameters:
K_steps, L_steps ==> correspond to K and L in BMG.
min_size_before_training ==> SAC needs to sample from the replay buffer, so this is the minimum buffer size before start training.
m_lr ==> corresponds to learning rate of the network of hyper-parameter since we choose to take hyper-parameter as output of some network and learn the network during the following training process.

# Mujoco_Ant
we conduct experiment based on an open-source RL code lab ILSwiss.
Check out ILSwiss at https://github.com/Ericonaldo/ILSwiss

If you want to conduct experiment on mujoco-ant, here's the command line: (Also make sure you conduct experiment on an environment that satisfy ILSwiss requirements!)
cd h_divergence_meta_learning/ILSwiss
python run_experiment -e /exp_specs/bmg/bmg_ant.yaml -g 0

You can check out Mujoco-Ant environment at https://www.gymlibrary.dev/environments/mujoco/ant/
You can also check out some parameters of the experiment in /exp_specs/bmg/bmg_ant.yaml, below are some demonstration of the parameters:
constants ==> parameters of policy, value function and BMG.
rl_alg_params ==> parameters of basic RL procedure, like parameters of sampling from environment
sac_params ==> parameters of SAC algorithm

You can check out experiment logs in h_divergence_meta_learning/ILSwiss/logs