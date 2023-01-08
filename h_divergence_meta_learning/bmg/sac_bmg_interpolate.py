import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
import torchopt as TorchOpt
import torchopt
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
from tcgw_env import TwoColorGridWorld
from craig import coreset_order

class Policy(nn.Module):
    def __init__(self, input_dims, naction_replay_buffer, lr, fc1_dims=256, fc2_dims=256, cuda_device_num = 0, gamma=0.99):
        super(Policy, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_sac_policy.pt')
        self.pi1 = nn.Linear(input_dims,fc1_dims)
        self.pi2 = nn.Linear(fc1_dims,fc2_dims)
        self.pi = nn.Linear(fc2_dims,naction_replay_buffer)

        self.device = T.device('cuda:'+str(cuda_device_num) if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optim = TorchOpt.MetaAdam(self,lr=lr, use_accelerated_op=True, moment_requires_grad=False)

    def forward(self, obs):

        pi = F.relu(self.pi1(obs))
        pi = F.relu(self.pi2(pi))
        pi = self.pi(pi)

        probs = T.softmax(pi, dim=-1)

        return probs

    def choose_action(self, observation):# return array shape = [1,]
        dist = Categorical(self.forward(observation))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def save_checkpoint(self):
        print("...Saving Policy Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Policy Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

        

class Value(nn.Module):
    def __init__(self, input_dims, lr, fc1_dims=256, fc2_dims=256, cuda_device_num = 0, gamma=0.99):
        super(Value, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_sac_value.pt')
        self.v1 = nn.Linear(input_dims,fc1_dims)
        self.v2 = nn.Linear(fc1_dims,fc2_dims)
        self.v = nn.Linear(fc2_dims,1)

        self.device = T.device('cuda:'+str(cuda_device_num) if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.optim = TorchOpt.MetaAdam(self,lr=lr, use_accelerated_op=True, moment_requires_grad=False)

    def forward(self, obs):

        v = F.relu(self.v1(obs))
        v = F.relu(self.v2(v))
        v = self.v(v)
        return v

    def save_checkpoint(self):
        print("...Saving Value Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Value Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

        

class Qfunc(nn.Module):
    def __init__(self, input_dims, naction_replay_buffer, lr, fc1_dims=256, fc2_dims=256, cuda_device_num = 0, gamma=0.99):
        super().__init__()
        self.chkpt_file = os.path.join("todo"+'_sac_Qfunc.pt')
        self.q1 = nn.Linear(input_dims,fc1_dims)
        self.q2 = nn.Linear(fc1_dims,fc2_dims)
        self.q = nn.Linear(fc2_dims,naction_replay_buffer)

    def forward(self, obs):
        q = F.relu(self.q1(obs))
        q = F.relu(self.q2(q))
        q = self.q(q)
        return q

    def save_checkpoint(self):
        print("...Saving Qfunc Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Qfunc Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class MetaMLP(nn.Module):
    def __init__(self, prior, lr, betas=(0.9, 0.999), eps=1e-4, input_dims=10, cuda_device_num = 0, fc1_dims=32):
        super(MetaMLP, self).__init__()
        self.chkpt_file = os.path.join("todo"+'_bmg')
        self.prior = prior
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, 1)

        self.optim = T.optim.Adam(self.parameters(),lr=lr, betas=betas, eps=eps)
        self.device = T.device('cuda:'+str(cuda_device_num) if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = T.sigmoid(self.fc2(out))
        return out + self.prior
    
    def save_checkpoint(self):
        print("...Saving Checkpoint...")
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print("...Loading Checkpoint...")
        self.load_state_dict(T.load(self.chkpt_file))

class Agent:
    def __init__(self, input_dims, naction_replay_buffer, gamma, prior, lr, m_lr, alpha, betas, eps, name, 
                    env, steps, K_steps, L_steps, rollout_steps_per_call, buffer_size, coreset_size, coreset_threshold, min_size_before_training, train_batch_size, discount, meta_start_epoch, cuda_device_num, random_seed):
        super(Agent, self).__init__()
        self.device = T.device('cuda:'+str(cuda_device_num) if T.cuda.is_available() else 'cpu')

        self.policy = Policy(input_dims, naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num)
        self.policy_k = Policy(input_dims, naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num)
        # self.vf = Value(input_dims, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num)
        self.qf1 = Qfunc(input_dims, naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num).to(self.device)
        self.qf2 = Qfunc(input_dims, naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num).to(self.device)
        self.meta_mlp = MetaMLP(prior, m_lr, betas, eps, input_dims=10, fc1_dims=32, cuda_device_num = cuda_device_num)

        self.qf1_optim = TorchOpt.MetaAdam(self.qf1,lr=lr, use_accelerated_op=True, moment_requires_grad=False)
        self.qf2_optim = TorchOpt.MetaAdam(self.qf2,lr=lr, use_accelerated_op=True, moment_requires_grad=False)

        self.buffer_size = buffer_size
        self.coreset_size = coreset_size
        self.coreset_threshold = coreset_threshold
        self.obs_replay_buffer = np.zeros((buffer_size, input_dims))
        self.next_obs_replay_buffer = np.zeros((buffer_size, input_dims))
        self.action_replay_buffer = np.zeros((buffer_size, 1))
        self.reward_replay_buffer = np.zeros((buffer_size, 1))
        self.min_size_before_training = min_size_before_training
        self.train_batch_size = train_batch_size
        self.effect_num = 0
        self.effect_ptr = 0
        self.meta_start_epoch = meta_start_epoch
        self.matching_loss = torch.nn.KLDivLoss(reduction='batchmean')
        

        self.env = env
        self.name = f"agent_{name}"
        self.naction_replay_buffer = naction_replay_buffer
        self.input_dims = input_dims
        self.steps = steps
        self.K_steps = K_steps
        self.L_steps = L_steps
        self.rollout_steps_per_call = rollout_steps_per_call
        self.random_seed = random_seed
        self.alpha = alpha
        self.gamma = gamma
        self.discount = discount
        
        #stats
        self.avg_reward = [0 for _ in range(10)]
        self.accum_reward = 0
        self.cum_reward = []
        self.epoch_reward = []
        self.entropy_rate = []

    def rollout(self):
        obs = torch.tensor(self.env.reset()).to(self.device, torch.float)
        rollout_reward = 0
        bootstrap_states = np.zeros((self.rollout_steps_per_call, self.input_dims))
        for idx in range(self.rollout_steps_per_call):
            action, _ = self.policy.choose_action(obs)
            obs = obs.cpu().numpy()
            action = action.cpu().numpy()
            obs_, reward, done, _ = self.env.step(action)
            
            if (self.effect_ptr - self.coreset_threshold) % (self.coreset_threshold + self.coreset_size)== 0:
                self.coreset_compress()
            
            self.obs_replay_buffer[self.effect_ptr] = obs
            bootstrap_states[idx] = obs
            self.next_obs_replay_buffer[self.effect_ptr] = obs_
            self.reward_replay_buffer[self.effect_ptr] = reward
            self.action_replay_buffer[self.effect_ptr] = action
            
            self.epoch_reward.append(reward)
            rollout_reward += reward
            self.accum_reward += reward
            self.cum_reward.append(self.accum_reward)

            obs = torch.tensor(obs_).to(self.device, torch.float)
            self.effect_ptr += 1
            self.effect_num += 1
            # No need, since non-episodic
            '''
            if done:
                break
            '''
        self.avg_reward = self.avg_reward[1:] 
        self.avg_reward.append(rollout_reward / self.rollout_steps_per_call)
        # print(np.sum(np.abs(rollout_reward)) / self.rollout_steps_per_call)
        ar = T.tensor(np.array(self.avg_reward)).to(self.policy.device, dtype=T.float)
        alpha = self.meta_mlp(ar)
        self.entropy_rate.append(alpha.item()) 
        return alpha, bootstrap_states

    def coreset_compress(self):
        print(f"start compress. size: {self.effect_ptr}")
        start_size = self.effect_ptr - self.coreset_threshold
        
        unit_size = 5000
        unit_select = int(unit_size * (self.coreset_size / self.coreset_threshold))
        unit_round = (self.coreset_threshold + unit_size - 1) // unit_size
        
        order = np.array([])
        for round in range(unit_round):
            min_idx = start_size + unit_size * round
            max_idx = min(start_size + unit_size * (round + 1), self.effect_ptr)
            X = np.concatenate((self.action_replay_buffer[min_idx : max_idx].repeat(20, axis=1), self.reward_replay_buffer[min_idx : max_idx].repeat(20, axis=1)), axis=1)
            X = np.concatenate((X, self.obs_replay_buffer[min_idx : max_idx], self.next_obs_replay_buffer[min_idx : max_idx]), axis=1)
                
            selected_order, _ = coreset_order(X, 'euclidean', unit_select, self.reward_replay_buffer[min_idx : max_idx].copy())
            order = np.append(order, selected_order)
        
        print(f'order type/shape: {type(order)} / {order.shape}')
        order = order.astype('int64')
        selectaction_replay_buffer = self.action_replay_buffer[order]
        selectreward_replay_buffer = self.reward_replay_buffer[order]
        selectobs_replay_buffer = self.obs_replay_buffer[order]
        selectnext_obs_replay_buffer = self.next_obs_replay_buffer[order]
        
        
        self.obs_replay_buffer[self.effect_ptr : self.effect_ptr + self.coreset_size] = selectobs_replay_buffer
        self.next_obs_replay_buffer[self.effect_ptr : self.effect_ptr + self.coreset_size] = selectnext_obs_replay_buffer
        self.action_replay_buffer[self.effect_ptr : self.effect_ptr + self.coreset_size] = selectaction_replay_buffer

        self.reward_replay_buffer[self.effect_ptr : self.effect_ptr + self.coreset_size] = selectreward_replay_buffer
        self.effect_ptr += self.coreset_size
        
        print(f"compress success without delete! size: {self.effect_ptr}")
        


    def train_step(self, epoch, meta_alpha, bootstrap=False):
        idx = np.random.choice(self.effect_ptr + 1, self.train_batch_size, replace=False)
        rewards = torch.tensor(self.reward_replay_buffer[idx]).to(self.device, dtype=torch.float)
        obs = torch.tensor(self.obs_replay_buffer[idx]).to(self.device, dtype=torch.float)
        actions = torch.tensor(self.action_replay_buffer[idx]).to(self.device, dtype=torch.float)
        next_obs = torch.tensor(self.next_obs_replay_buffer[idx]).to(self.device, dtype=torch.float)
        next_action_dist = Categorical(self.policy(next_obs))
        """
        QF Loss
        """
        if ~bootstrap:
            q1_pred = self.qf1(obs).gather(1, actions.to(dtype=int))
            q2_pred = self.qf2(obs).gather(1, actions.to(dtype=int))
            next_q = next_action_dist.probs * torch.min(
            self.qf1(next_obs),
            self.qf2(next_obs),
            )
            if epoch >= self.meta_start_epoch:
                target_v_values = next_q.sum(dim=-1) + meta_alpha * next_action_dist.entropy()
            else:
                target_v_values = next_q.sum(dim=-1) + self.alpha * next_action_dist.entropy()
            q_target = rewards + self.discount * target_v_values.unsqueeze(-1)
            q_target = q_target.detach()
            qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
            qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

            self.qf1_optim.step(qf1_loss)
            self.qf2_optim.step(qf2_loss)

        """
        Policy Loss
        """
        qf1_copy = Qfunc(input_dims, self.naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num).to(self.device)
        qf2_copy = Qfunc(input_dims, self.naction_replay_buffer, lr, fc1_dims = 256, fc2_dims = 256, cuda_device_num = cuda_device_num).to(self.device)
        qf1_copy.load_state_dict(self.qf1.state_dict())
        qf2_copy.load_state_dict(self.qf2.state_dict())
        q1_new_acts = qf1_copy(obs)
        q2_new_acts = qf2_copy(obs)  ## error
        q_newaction_replay_buffer = torch.min(q1_new_acts, q2_new_acts)

        action_dist = Categorical(self.policy(obs))
        if ~bootstrap and epoch >= self.meta_start_epoch:
            policy_loss = -torch.mean(meta_alpha * action_dist.entropy() + (action_dist.probs * q_newaction_replay_buffer).sum(dim=-1))
        else:
            policy_loss = -torch.mean(self.alpha * action_dist.entropy() + (action_dist.probs * q_newaction_replay_buffer).sum(dim=-1))
        self.policy.optim.step(policy_loss)

        return torch.mean(policy_loss).item(), torch.mean(q1_new_acts).item(), torch.mean(q2_new_acts).item()

    def matching_function(self, policy_k, tb, states, policy_k_state_dict):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        with T.no_grad():
            dist_tb = tb(states)

        TorchOpt.recover_state_dict(policy_k, policy_k_state_dict)
        dist_k = policy_k(states)
        
        # KL Div between dsitributions of TB and AC_K, respectively
        kl_div = self.matching_loss(dist_k, dist_tb)

        return kl_div

    def plot_results(self):

        cr = plt.figure(figsize=(10, 10)) 
        plt.plot(self.cum_reward)
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.savefig('res/cumulative_reward')
        plt.close(cr)

        er = plt.figure(figsize=(10, 10))
        plt.plot(list(range(2e4)), self.entropy_rate[int(-2e4):])
        plt.xlabel('Last 20,000 alphas')
        plt.ylabel('Entropy Rate')
        plt.savefig('res/entropy_rate')
        plt.close(er)

    def run(self):
        while self.effect_num <= self.min_size_before_training:
            self.rollout()
        outer_range = self.steps // self.rollout_steps_per_call
        ct = 0
        self.policy_loss = []
        self.q1_mean = []
        self.q2_mean = []
        for epoch in range(outer_range):
            alpha, bootstrap_states = self.rollout()
            for _ in range(self.K_steps):
                self.train_step(epoch, alpha)
            k_state_dict = TorchOpt.extract_state_dict(self.policy)
            
            for _ in range(self.L_steps-1):
                policy_loss, q1_mean, q2_mean = self.train_step(epoch, alpha)
            k_l_m1_state_dict = TorchOpt.extract_state_dict(self.policy)
            k_l_m1_optim_dict = TorchOpt.extract_state_dict(self.policy.optim)

            self.train_step(epoch, alpha, bootstrap=True)

            # KL-Div Matching loss
            matching_loss = self.matching_function(self.policy_k, self.policy, bootstrap_states, k_state_dict)
            
            # MetaMLP update
            self.meta_mlp.optim.zero_grad()
            matching_loss.backward()
            self.meta_mlp.optim.step()

            # Use most recent params and stop grad
            TorchOpt.recover_state_dict(self.policy, k_l_m1_state_dict)
            TorchOpt.recover_state_dict(self.policy.optim, k_l_m1_optim_dict)
            TorchOpt.stop_gradient(self.policy)
            TorchOpt.stop_gradient(self.policy.optim)
            TorchOpt.stop_gradient(self.qf1)
            TorchOpt.stop_gradient(self.qf1_optim)
            TorchOpt.stop_gradient(self.qf2)
            TorchOpt.stop_gradient(self.qf2_optim)

            ct += self.rollout_steps_per_call

            # print stats
            if ct %10000 == 0:
                print(f"CR and ER, step# {ct}:")
                print(self.cum_reward[-1])
                print(self.entropy_rate[-1])
                print("reward mean:", np.mean(np.array(self.epoch_reward)))
                print("reward_max:", np.max(np.array(self.epoch_reward)))
                print("reward_min:", np.min(np.array(self.epoch_reward)))
                self.epoch_reward.clear()
                print("policy loss:", policy_loss)
                print("q1 mean", q1_mean)
                print("q2 mean", q2_mean)
                print("###")

if __name__ == "__main__":
    '''Driver code'''
    steps = 4_800_000
    K_steps = 5
    L_steps = 5
    rollout_steps_per_call = 50
    train_batchsize = 256
    buffer_size = 4800000
    coreset_size = 100000
    coreset_threshold = 500000
    min_size_before_training = 1000
    discount = 0.99
    meta_start_epoch = 0
    random_seed = 5
    env = TwoColorGridWorld()
    naction_replay_buffer = 4
    input_dims = env.observation_space.shape[0]
    gamma = 0.99
    lr = 1e-4
    m_lr = 1e-4
    alpha = 0.05
    prior = 0
    betas = (0.9, 0.999)
    eps = 1e-4
    cuda_device_num = 1
    name = 'meta_agent_bmg' 

    # set seed
    T.cuda.manual_seed(random_seed)
    T.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent = Agent(input_dims, naction_replay_buffer, gamma, prior, lr, m_lr, alpha, betas, eps, name, env, 
                    steps, K_steps, L_steps, rollout_steps_per_call, buffer_size, coreset_size, coreset_threshold, min_size_before_training, 
                    train_batchsize, discount, meta_start_epoch, cuda_device_num, random_seed)
    agent.run()
    print("done")
    agent.plot_results()
