"""

jsrl.py

Implementation of Jump-Start Reinforcement Learning (JSRL) with various training algorithms

"""

import sys
import time

from typing import Union, Optional, List
from pathlib import Path
from collections import deque

import numpy as np
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import onnxruntime as ort


class JSSAC(SAC):
    """ SAC + Jump-Start
    Start by letting the guide_policy decide all actions to take.
    Calculate the reward and check if it is above the reward_threshold.
    If yes, decrease the number of actions coming from the guide_policy.
    If not, collect more data.
    Do this until the SACPolicy is acting alone

    :param guide_model_path: The path to the guide model .onnx
    :param max_env_steps: The maximum number of steps that the environment can take before
        resetting
    :param n_curricula: The number of curricula (= steps until the exploration policy takes
        over)
    :param reward_threshold: If this threshold is crossed, the next curriculum starts.
        Given as a difference in % to the mean best reward achieved during curriculum 1.
    :param sac_kwargs: Keyworded args for the SAC class
    """

    def __init__(self, policy, env, guide_model_path: Union[str, Path], n_reward_mean: int, n_curricula: int,
                 reward_threshold: float, sac_kwargs: dict, verbose: int = 1):

        # load the guide_policy, it must be trained with SAC in the same environment
        try:
            self.guide_model = ort.InferenceSession(guide_model_path)
        except:
            print(f'Could not load guide model from path {guide_model_path}!')
            #raise ValueError(f'Could not load guide policy from path {guide_model_path}!')

        self.n_curricula = n_curricula
        self.reward_threshold = reward_threshold
        self.max_env_steps = env.unwrapped.max_episode_length
        self.current_curriculum = 0
        self.done_envs = np.zeros(env.num_envs, dtype=bool)
        self.reward_achieved = False
        self.reward_mean_buf = deque(maxlen=n_reward_mean)
        self.reward_baseline = None

        # init SAC
        super().__init__(policy, env, verbose=verbose, **sac_kwargs)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        # JSRL specific logging
        self.logger.record("jsrl/current_curriculum", self.current_curriculum)
        self.logger.record("jsrl/reward_mean", np.mean(self.reward_mean_buf))
        if self.reward_baseline is not None:
            self.logger.record("jsrl/reward_threshold", self.reward_baseline * self.reward_threshold)

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # check, which policy should issue the action: guide_policy or exploration_policy
            # if the number of steps < max_env_steps - (max_env_steps / current_curriculum)
            # -> guide_policy, otherwise exploration_policy

            # Select action randomly or according to policy
            actions_unguided, buffer_actions_unguided = self._sample_action(learning_starts, action_noise, env.num_envs)

            # create onnx model input; the env needs to have a method called _get_guide_model_obs
            actions_guided = []
            for env_idx in range(env.num_envs):
                model_inputs = {self.guide_model.get_inputs()[0].name: self.env.unwrapped._get_guide_model_obs().cpu().numpy()[env_idx].reshape(1, -1)}
                actions_guided.append(self.guide_model.run(None, model_inputs)[0])

            actions_guided = np.concatenate(actions_guided, axis=0)
            buffer_actions_guided = actions_guided

            envs_guide_policy_cond = self.env.unwrapped.time_steps.cpu().numpy() < self.max_env_steps - self.current_curriculum * (self.max_env_steps / self.n_curricula)
            actions = np.where(envs_guide_policy_cond.reshape(env.num_envs, env.action_space.shape[0]), actions_guided, actions_unguided)
            buffer_actions = np.where(envs_guide_policy_cond.reshape(env.num_envs, env.action_space.shape[0]), buffer_actions_guided, buffer_actions_unguided)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # add the latest total reward to the reward_mean buffer
                    self.reward_mean_buf.append(self.ep_info_buffer[-1]['r'])

                    # check if we have already reached the required number of episodes to calculate the reward mean
                    if len(self.reward_mean_buf) == self.reward_mean_buf.maxlen:
                        reward_mean = np.mean(self.reward_mean_buf)
                        # if it is the first curriculum, calculate the reward baseline
                        if self.current_curriculum == 0:
                            self.reward_baseline = reward_mean

                    self.done_envs[idx] = True

            # check if the reward_mean is above the reward_threshold
            if self.reward_baseline is not None and reward_mean >= self.reward_baseline * self.reward_threshold:
                self.reward_achieved = True

            # check if all environments are done and the reward was achieved to start the next curriculum
            if self.done_envs.all() and self.reward_achieved:
                self.current_curriculum += 1
                self.done_envs = np.zeros(env.num_envs, dtype=bool)
                self.reward_achieved = False

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["guide_model"]