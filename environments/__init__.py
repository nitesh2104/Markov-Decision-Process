import gym
from gym.envs.registration import register

from .frozen_lake import *


__all__ = ['RewardingFrozenLakeEnv']

# FrozenLake
register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)


def get_frozen_lake_environment():
    return gym.make('FrozenLake-v0')


register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': True, 'is_slippery': False}
)


def get_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake8x8-v0')


register(
    id='RewardingandSlipperyFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': True, 'is_slippery': True}
)


def get_small_rewarding_and_slipper_frozen_lake_environment():
    return gym.make('RewardingandSlipperyFrozenLake8x8-v0')


register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)


def get_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')


register(
    id='RewardingFrozenLakeWithRewards15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15', 'rewarding': True, 'is_slippery': False}
)


def get_rewarding_medium_frozen_lake():
    return gym.make('RewardingFrozenLakeWithRewards15x15-v0')


register(
    id='FrozenLakeWithoutRewardsAndIsSlippery15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15', 'rewarding': True, 'is_slippery': True}
)


def get_medium_frozen_lake_without_rewards_and_slippery():
    return gym.make('FrozenLakeWithoutRewardsAndIsSlippery15x15-v0')


register(
    id='RewardingLargeFrozenLakeRewardsAndSlippery20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': True, 'is_slippery': False}
)


def get_large_with_rewards_and_slippery():
    return gym.make('RewardingLargeFrozenLakeRewardsAndSlippery20x20-v0')


register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)


def get_rewarding_large_frozen_lake():
    return gym.make('RewardingFrozenLakeWithRewards20x20-v0')


register(
    id='RewardingFrozenLakeWithRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': True, 'is_slippery': False}
)


def get_large_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')
