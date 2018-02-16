from mcts_utils import GameState, A
from typing import Callable, Union
from numpy.random import RandomState


class RandomKStepRollOut(object):
    """
    Estimate the reward with the sum of returns of a k step rollout
    """

    def __init__(self, k: int, random_state: Union[RandomState, None] = None):
        self._k = k
        self._random_state = random_state or RandomState()

    def __call__(self, state: GameState) -> float:
        current_k: float = 0
        total_rewards: float = 0.0

        while not (state.is_terminal() or current_k > self._k):
            action: A = self._random_state.choice(state.actions)
            state = state.perform(action=action)
            total_rewards += state.reward()
            current_k += 1

        return total_rewards
