from collections import OrderedDict
import torch
import numpy as np
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm


class TorchMetaRLAlgorithm(TorchBaseAlgorithm):
    def __init__(
        self, trainer, batch_size, num_train_steps_per_train_call, inner_loop_steps, bootstrap_loop_steps, device, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.batch_size = batch_size
        self.num_train_steps_per_train_call = num_train_steps_per_train_call
        self.inner_train_steps_total = 0
        self.inner_loop_steps = inner_loop_steps
        self.bootstrap_loop_steps = bootstrap_loop_steps
        self.device = device

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    def networks(self):
        return self.trainer.networks

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def _do_training(self, epoch):
        all_reward_per_iter = np.sort(np.concatenate(self.reward_list_one_iter, axis=1))
        avg_reward_per_iter = torch.zeros(self.inference_reward_num + 1).to(self.device)
        interval = self.num_env_steps_per_epoch / self.inference_reward_num
        for i in range(self.inference_reward_num):
            if i == self.inference_reward_num - 1:
                avg_reward_per_iter[i] = np.mean(all_reward_per_iter[i*interval: ])
            else:
                avg_reward_per_iter[i] = np.mean(all_reward_per_iter[i*interval: (i+1)*interval])
        for step in range(self.num_train_steps_per_train_call):
            self.inner_train_steps_total = step + self._n_train_steps_total*self.num_train_steps_per_train_call
            avg_reward_per_iter[-1] = step / self.num_train_steps_per_train_call #normalization for step
            if getattr(self.trainer, "on_policy", False):
                self.trainer.train_step(self.get_all_trajs(), self.inner_train_steps_total, avg_reward_per_iter)
                self.clear_buffer()
            else:
                self.trainer.train_step(self.get_batch(), self.inner_train_steps_total, avg_reward_per_iter)
        self.reward_list_one_iter.clear()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(epoch=epoch)
        data_to_save.update(self.trainer.get_snapshot())
        return data_to_save

    def load_snapshot(self, snapshot):
        self.trainer.load_snapshot(snapshot)
        self.exploration_policy = self.trainer.policy
        from rlkit.torch.common.policies import MakeDeterministic

        self.eval_policy = MakeDeterministic(self.exploration_policy)
        self.eval_sampler.policy = self.eval_policy

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        trainer_stats = self.trainer.get_eval_statistics()
        if trainer_stats is not None:
            self.eval_statistics.update(trainer_stats)
        super().evaluate(epoch)

    def _end_epoch(self):
        self.trainer.end_epoch()
        super()._end_epoch()

    def get_all_trajs(self):
        batch = self.replay_buffer.sample_all_trajs()
        batch = [np_to_pytorch_batch(b) for b in batch]
        return batch

    def clear_buffer(self):
        self.replay_buffer.clear()
