import yaml
import argparse
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
import torch
from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.logger import load_from_file
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp, Mlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.bmg.bmg import BootstrappedMetaGradient
from rlkit.torch.algorithms.torch_meta_rl_algorithm import TorchMetaRLAlgorithm


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env_wrapper = ProxyEnv  # Identical wrapper
    wrapper_kwargs = {}

    if isinstance(act_space, gym.spaces.Box):
        env_wrapper = NormalizedBoxEnv

    env = env_wrapper(env, **wrapper_kwargs)

    kwargs = {}
    if "vec_env_kwargs" in env_specs:
        kwargs = env_specs["env_kwargs"]["vec_env_kwargs"]

    training_env = get_envs(env_specs, env_wrapper, wrapper_kwargs, **kwargs)
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    net_size = variant["net_size"]
    meta_net_size = variant["meta_net_size"]
    inference_reward_num = variant["inference_reward_num"]
    num_hidden = variant["num_hidden_layers"]
    inner_loop_steps = variant["inner_loop_steps"]
    bootstrap_loop_steps = variant["bootstrap_loop_steps"]
    matching_loss = variant["matching_loss"]
    matching_mean_coef = variant["matching_mean_coef"]
    matching_std_coef = variant["matching_std_coef"]
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    meta_mlp = Mlp(
        hidden_sizes=num_hidden * [meta_net_size],
        input_size=inference_reward_num + 1,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    policy_k = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    trainer = BootstrappedMetaGradient(
        policy=policy, policy_k=policy_k, meta_mlp=meta_mlp,
        qf1=qf1, qf2=qf2, vf=vf, inner_loop_steps=inner_loop_steps, bootstrap_loop_steps=bootstrap_loop_steps, 
        matching_mean_coef=matching_mean_coef, matching_std_coef=matching_std_coef, matching_loss=matching_loss, **variant["sac_params"]
    )
    algorithm = TorchMetaRLAlgorithm(
        trainer=trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        inner_loop_steps=inner_loop_steps,
        bootstrap_loop_steps=bootstrap_loop_steps,
        inference_reward_num=inference_reward_num,
        device=ptu.device,
        **variant["rl_alg_params"]
    )
    trainer.meta_observations = torch.zeros((trainer.num_steps_per_loop, algorithm.batch_size, algorithm.replay_buffer._observation_dim))

    epoch = 0
    if "load_params" in variant:
        algorithm, epoch = load_from_file(algorithm, **variant["load_params"])

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    algorithm.train(start_epoch=epoch)

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    log_dir = None
    if "load_params" in exp_specs:
        load_path = exp_specs["load_params"]["load_path"]
        if (load_path is not None) and (len(load_path) > 0):
            log_dir = load_path

    setup_logger(
        exp_prefix=exp_prefix,
        exp_id=exp_id,
        variant=exp_specs,
        seed=seed,
        log_dir=log_dir,
    )

    experiment(exp_specs)
