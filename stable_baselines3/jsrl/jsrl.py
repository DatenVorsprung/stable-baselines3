"""

jsrl.py

Implementation of Jump-Start Reinforcement Learning (JSRL) with various training algorithms

"""
from typing import Union, Optional
from pathlib import Path

from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv


class JSSAC(SAC):
    """ SAC + Jump-Start
    Start by letting the guide_policy decide all actions to take.
    Calculate the reward and check if it is above the reward_threshold.
    If yes, decrease the number of actions coming from the guide_policy.
    If not, collect more data.
    Do this until the SACPolicy is acting alone

    :param guide_policy_path: The path to the guide policy .zip
    :param guide_policy_obs_key: The key in the environments observation dict, that the guide
        policy uses for predicting actions
    :param max_env_steps: The maximum number of steps that the environment can take before
        resetting
    :param n_curricula: The number of curricula (= steps until the exploration policy takes
        over)
    :param reward_threshold: If this threshold is crossed, the next curriculum starts.
        Given as a difference in % to the mean best reward achieved during curriculum 1.
    :param sac_kwargs: Keyworded args for the SAC class
    """

    def __init__(self, guide_policy_path: Union[str, Path], guide_policy_obs_key: str,
                 max_env_steps: int, n_curricula: int, reward_threshold: float, sac_kwargs: dict):

        # load the guide_policy, it must be trained with SAC in the same environment
        try:
            self.guide_policy = SAC.load(guide_policy_path, sac_kwargs['env'])
        except:
            raise ValueError(f'Could not load guide policy from path {guide_policy_path}!')

        self.n_curricula = n_curricula
        self.guide_policy_obs_key = guide_policy_obs_key
        self.reward_threshold = reward_threshold
        self.max_env_steps = max_env_steps
        self.current_curriculum = 1

        # init SAC
        super().__init__(**sac_kwargs)


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
            if num_collected_steps

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

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
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
