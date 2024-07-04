"""

jsrl.py

Implementation of Jump-Start Reinforcement Learning (JSRL) with various training algorithms

"""
from typing import Union

from pathlib import Path
from stable_baselines3.sac.sac import SAC


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
    :param n_curricula: The number of curricula (= steps until the exploration policy takes
        over)
    :param reward_threshold: If this threshold is crossed, the next curriculum starts.
        Given as a difference in % to the mean best reward achieved during curriculum 1.
    :param sac_kwargs: Keyworded args for the SAC class
    """

    def __init__(self, guide_policy_path: Union[str, Path], guide_policy_obs_key: str,
                 n_curricula: int, reward_threshold: float, sac_kwargs: dict):

        # load the guide_policy, it must trained with SAC in the same environment
        try:
            self.guide_policy = SAC.load(guide_policy_path, sac_kwargs['env'])
        except:
            raise ValueError(f'Could not load guide policy from path {guide_policy_path}!')

        self.n_curricula = n_curricula
        self.guide_policy_obs_key = guide_policy_obs_key
        self.reward_threshold = reward_threshold

        # init SAC
        super().__init__(**sac_kwargs)


