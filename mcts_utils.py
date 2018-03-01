from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Union
from multiprocessing import Pipe
import numpy

DEBUG = False
A = TypeVar('Action')


class GameState(ABC, Generic[A]):
    """GameState is an immutable class that represent the state of the game"""

    def __init__(self, **kwargs):
        for key in kwargs:
            object.__setattr__(self, key, kwargs[key])

    @abstractmethod
    def perform(self, action: A) -> 'GameState':
        """returns a new state object, with the newly created state"""
        raise NotImplementedError

    @abstractmethod
    def reward(self) -> float:
        """return the reward for taking action from this state"""
        raise NotImplementedError

    @property
    @abstractmethod
    def actions(self) -> List[A]:
        """returns the available actions to the agent from this state"""
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        """returns True if this state is terminal"""
        raise NotImplementedError

    def __setattr__(self, *args):
        raise TypeError('GameState is immutable!')

    def __delattr__(self, *args):
        raise TypeError('GameState is immutable!')


class StateNode(Generic[A]):
    """StateNode represents a state node in the tree"""

    def __init__(self, parent: Union['StateNode', None], action: A, game_state: GameState, round_n: int = 0):
        self._children: Dict[A, StateNode] = {}
        self._game_state = game_state
        self._parent = parent
        self._action = action
        self._depth = parent._depth + 1 if parent else 0
        self._round_n = round_n
        self._value = 0.0
        self._visits = 0

    def add_child(self, child: A, value: 'StateNode') -> None:
        self.children[child] = value

    def make_root(self):
        self._parent = None

    @property
    def round_n(self) -> int:
        return self._round_n

    @property
    def state(self) -> GameState:
        return self._game_state

    @property
    def children(self) -> Dict[A, 'StateNode']:
        return self._children

    @property
    def visits(self):
        return self._visits

    @visits.setter
    def visits(self, new_value: int) -> None:
        self._visits = new_value

    @property
    def untried_actions(self) -> List[A]:
        return [action for action in self.game_state.actions if self.children.get(action, None) is None]

    @property
    def game_state(self) -> GameState:
        return self._game_state

    def refresh_state(self, round_n: int, game_state: GameState = None):
        self._game_state = game_state or \
                           self.parent.game_state.perform(self.action)
        self._round_n = round_n

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        self._value = new_value

    @property
    def parent(self) -> 'StateNode':
        return self._parent

    @property
    def action(self) -> A:
        return self._action

    @property
    def depth(self) -> int:
        return self._depth


class Worker:
    """Worker is a worker node, used for multiprocessing"""

    def __init__(self, pipe: 'Pipe', random_seed: int, time_limit=float('inf')):
        self._pipe = pipe
        self._random_state = numpy.random.RandomState(random_seed)
        self._time_limit = time_limit
