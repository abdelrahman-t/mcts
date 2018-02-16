import numpy
from mcts_utils import StateNode


class UCB1(object):
    """
    The typical bandit upper confidence bounds algorithm.
    """

    def __init__(self, c: float):
        self.c = c

    def __call__(self, state: StateNode) -> float:
        return (state.value / state.visits) + \
               numpy.sqrt((2 * numpy.log(state.parent.visits) / state.visits)) * self.c
