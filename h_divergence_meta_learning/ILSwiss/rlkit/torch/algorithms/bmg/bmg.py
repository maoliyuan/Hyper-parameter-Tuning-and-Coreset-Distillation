from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim
import TorchOpt
import copy
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class BootstrappedMetaGradient(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    TODO: Recently in rlkit there is a version which only uses two Q functions
    as well as an implementation of entropy tuning but I have not implemented
    those
    """

    def __init__(
        self,
        policy: nn.Module,
        policy_k: nn.Module,
        qf1: nn.Module,
        qf2: nn.Module,
        vf: nn.Module,
        meta_mlp: nn.Module,
        reward_scale=1.0,
        discount=0.99,
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        meta_mlp_lr=1e-4,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        inner_loop_steps=1, 
        bootstrap_loop_steps=1,
        optimizer_class=optim.Adam,
        meta_optimizer_class=TorchOpt.MetaAdam,
        matching_loss="mse",
        matching_mean_coef=1,
        matching_std_coef=1,
        beta_1=0.9,
        **kwargs,
    ):
        self.policy = policy
        self.policy_k = policy_k
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.meta_mlp = meta_mlp
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.inner_loop_steps = inner_loop_steps
        self.bootstrap_loop_steps = bootstrap_loop_steps
        self.num_steps_per_loop = inner_loop_steps + bootstrap_loop_steps
        self.meta_observations = None
        self.k_state_dict = None
        self.k_l_m1_state_dict = None
        self.matching_mean_coef = matching_mean_coef
        self.matching_std_coef = matching_std_coef
        if matching_loss == "mse":
            self.matching_loss = nn.MSELoss()

        self.target_vf = vf.copy()
        self.eval_statistics = None

        self.policy_optimizer = meta_optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = meta_optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = meta_optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.vf_optimizer = meta_optimizer_class(
            self.vf.parameters(), lr=vf_lr, betas=(beta_1, 0.999)
        )
        self.meta_mlp_optimizer = optimizer_class(
            self.meta_mlp.parameters(), lr=meta_mlp_lr, betas=(beta_1, 0.999)
        )

    def train_step(self, batch, n_train_step_total, avg_reward_per_iter):
        # q_params = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        # v_params = itertools.chain(self.vf.parameters())
        # policy_params = itertools.chain(self.policy.parameters())

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        self.meta_observations[n_train_step_total % self.num_steps_per_loop] = obs
        policy_outputs = self.policy(obs, return_log_prob=True)
        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        if n_train_step_total % self.num_steps_per_loop != self.num_steps_per_loop-1:
            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)
            target_v_values = self.target_vf(
                next_obs
            )  # do not need grad || it's the shared part of two calculation
            q_target = (
                rewards + (1.0 - terminals) * self.discount * target_v_values
            )  ## original implementation has detach
            q_target = q_target.detach()
            qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
            qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

            # freeze parameter of Q
            # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
            #     p.requires_grad = False
            """
            VF Loss
            """
            # Only unfreeze parameter of V
            # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
            #     p.requires_grad = False
            # for p in self.vf.parameters():
            #     p.requires_grad = True
            # for p in self.policy.parameters():
            #     p.requires_grad = True  ##
            v_pred = self.vf(obs)
            # Make sure policy accounts for squashing functions like tanh correctly!
            # in this part, we only need new_actions and log_pi with no grad
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            q1_new_acts = self.qf1(obs, new_actions)
            q2_new_acts = self.qf2(obs, new_actions)  ## error
            q_new_actions = torch.min(q1_new_acts, q2_new_acts)
            v_target = q_new_actions - self.meta_mlp(avg_reward_per_iter) * log_pi
            v_target = v_target.detach()
            vf_loss = 0.5 * torch.mean((v_pred - v_target) ** 2)

            self.qf1_optimizer.step(qf1_loss)
            self.qf2_optimizer.step(qf2_loss)

            self.vf_optimizer.step(vf_loss)
            """
            Update networks
            """
            # unfreeze all -> initial states
            # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
            #     p.requires_grad = True
            # for p in self.vf.parameters():
            #     p.requires_grad = True
            # for p in self.policy.parameters():
            #     p.requires_grad = True

            # unfreeze parameter of Q
            # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
            #     p.requires_grad = True

            self._update_target_network()

        """
        Policy Loss
        """
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False
        # for p in self.vf.parameters():
        #     p.requires_grad = False
        # for p in self.policy.parameters():
        #     p.requires_grad = True
        qf1_copy = copy.deepcopy(self.qf1)
        qf2_copy = copy.deepcopy(self.qf2)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = qf1_copy(obs, new_actions)
        q2_new_acts = qf2_copy(obs, new_actions)  ## error
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)

        if n_train_step_total % self.num_steps_per_loop != self.num_steps_per_loop-1:
            policy_loss = torch.mean(self.meta_mlp(avg_reward_per_iter) * log_pi - q_new_actions)
        else:
            policy_loss = torch.mean(-1 * q_new_actions)  ##
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        self.policy_optimizer.step()

        if n_train_step_total % self.num_steps_per_loop == self.inner_loop_steps-1:
            assert self.k_state_dict == None
            self.k_state_dict = TorchOpt.extract_state_dict(self.policy)
        if n_train_step_total % self.num_steps_per_loop == self.num_steps_per_loop-2:
            assert self.k_l_m1_state_dict == None
            self.k_l_m1_state_dict = TorchOpt.extract_state_dict(self.policy)
        if n_train_step_total % self.num_steps_per_loop == self.num_steps_per_loop-1:
            matching_loss = self.matching_function(self.policy_k, self.policy, self.meta_observations, self.k_state_dict)
            self.meta_mlp_optimizer.zero_grad()
            matching_loss.backward()
            self.meta_mlp_optimizer.step()
            TorchOpt.recover_state_dict(self.policy, self.k_l_m1_state_dict)
            TorchOpt.stop_gradient(self.policy)
            TorchOpt.stop_gradient(self.policy_optimizer)
            TorchOpt.stop_gradient(self.qf1)
            TorchOpt.stop_gradient(self.qf1_optimizer)
            TorchOpt.stop_gradient(self.qf2)
            TorchOpt.stop_gradient(self.qf2_optimizer)
            TorchOpt.stop_gradient(self.vf)
            TorchOpt.stop_gradient(self.vf_optimizer)
            self.k_state_dict = None
            self.k_l_m1_state_dict = None
        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["VF Loss"] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "V Predictions",
                    ptu.get_numpy(v_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy mu",
                    ptu.get_numpy(policy_mean),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.vf,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def matching_function(self, policy_k, tb, meta_observations, policy_k_state_dict):
        meta_observations = torch.reshape(meta_observations, (-1, meta_observations.shape[-1]))
        with torch.no_grad():
            policy_outputs_tb = tb(meta_observations)
            policy_mean_tb, policy_log_std_tb = policy_outputs_tb[1], policy_outputs_tb[2]
        
        TorchOpt.recover_state_dict(policy_k, policy_k_state_dict)
        policy_outputs_k = policy_k(meta_observations)
        policy_mean_k, policy_log_std_k = policy_outputs_k[1], policy_outputs_k[2]
        
        # Div between dsitributions of TB and AC_K, respectively
        div = self.matching_mean_coef * self.matching_loss(policy_mean_tb, policy_mean_k) + self.matching_std_coef * self.matching_loss(policy_log_std_tb, policy_log_std_k)

        return div

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
            vf_optimizer=self.vf_optimizer,
        )

    def load_snapshot(self, snapshot):
        self.qf1 = snapshot["qf1"]
        self.qf2 = snapshot["qf2"]
        self.policy = snapshot["policy"]
        self.vf = snapshot["vf"]
        self.target_vf = snapshot["target_vf"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.qf1_optimizer = snapshot["qf1_optimizer"]
        self.qf2_optimizer = snapshot["qf2_optimizer"]
        self.vf_optimizer = snapshot["self.vf_optimizer"]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
