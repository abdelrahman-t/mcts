import numpy
from numpy.random import RandomState

from time import sleep, time
from collections import Counter
from itertools import count

from multiprocess import Pool, Pipe, Process

from mcts_utils import StateNode, GameState, A
from default_policy import RandomKStepRollOut
from typing import Generator, Callable, AnyStr, List

"""
IEEE TRANSACTIONS ON COMPUTATIONAL INTELLIGENCE AND AI IN GAMES, VOL. 4, NO. 1, MARCH 2012
A Survey of Monte Carlo Tree Search Methods

Cameron Browne, Member, IEEE, Edward Powley, Member, IEEE, Daniel Whitehouse, Member, IEEE,
Simon Lucas, Senior Member, IEEE, Peter I. Cowling, Member, IEEE, Philipp Rohlfshagen,
Stephen Tavener, Diego Perez, Spyridon Samothrakis and Simon Colton

Page 9
"""
SUPPORTED_DEFAULT_POLICIES = {'random-k'}
SUPPORTED_TREE_POLICIES = {'ucb1'}
SUPPORTED_BACKUPS = {'monte-carlo'}


class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """

    def __init__(self, tree_policy: Callable, default_policy: Callable, backup: Callable, time_limit: float = 1.0,
                 max_iter: int = int(1e10)):
        self._tree_policy = tree_policy
        self._default_policy = default_policy
        self._backup = backup
        self._time_limit = time_limit
        self._max_iter = max_iter

    def uct_search(self, state: GameState) -> A:
        if state.is_terminal():
            raise Exception("Can't start from a terminal state!")

        root: StateNode = StateNode(parent=None, action=None, game_state=state)

        start_time = time()
        counter: Generator = count()

        while time() - start_time < self._time_limit and next(counter) < self._max_iter:
            state: StateNode = self._get_next_node(state=root)
            reward: float = self._default_policy(state.game_state)
            self._backup(state=state, reward=reward)

        return self._best_child(state=root, policy=lambda s: s.value / s.visits).action

    def _get_next_node(self, state: StateNode) -> StateNode:
        while not state.state.is_terminal():
            if state.untried_actions:
                return self._expand(state=state)
            else:
                state = self._best_child(state=state, policy=self._tree_policy)

        return state

    @staticmethod
    def _expand(state: StateNode) -> StateNode:
        action: A = numpy.random.choice(state.untried_actions)
        assert state.children[action] is None

        state.add_child(action, StateNode(parent=state, action=action,
                                          game_state=state.game_state.perform(action)))

        return state.children[action]

    @staticmethod
    def _best_child(state: StateNode, policy: Callable) -> StateNode:
        return max(state.children.values(), key=policy)


class MCTSRootParallel:
    """ Root parallel Monte carlo tree search with majority voting aka Ensemble UCT
        Y. Soejima, A. Kishimoto, and O. Watanabe, “Evaluating Root
        Parallelization in Go,” IEEE Trans. Comp. Intell. AI Games, vol. 2,
        no. 4, pp. 278–287, 2010.
    """

    def __init__(self, number_of_processes: int, tree_policy: Callable, default_policy: AnyStr,
                 backup: Callable, time_limit: float = 1.0, k: int = 1, max_iter: int = int(1e10)):

        self._pool = Pool(number_of_processes)
        self._pipes = [Pipe() for _ in range(number_of_processes)]
        self._number_of_processes = number_of_processes

        self._tree_policy = tree_policy
        self._default_policy = default_policy
        self._backup = backup
        self._time_limit = time_limit
        self._max_iter = max_iter
        self._k = k

        agents: List[_MCTSRootParallel] = []

        for process_number in range(self._number_of_processes):
            default_policy = RandomKStepRollOut(k=self._k, random_state=RandomState(process_number))

            agent = _MCTSRootParallel(pipe=self._pipes[process_number][1], tree_policy=self._tree_policy,
                                      default_policy=default_policy, backup=self._backup, time_limit=self._time_limit,
                                      max_iter=self._max_iter)

            self._pool.apply_async(agent.uct_search_with_pipes)
            agents.append(agent)

        self._agents = agents
        sleep(5)

    def run(self, state: GameState) -> A:
        for process in range(self._number_of_processes):
            self._pipes[process][0].send(state)

        actions = Counter([self._pipes[process][0].recv() for process in range(self._number_of_processes)])
        return actions.most_common(1)[0][0]


class _MCTSRootParallel(MCTS):
    def __init__(self, pipe: Pipe, tree_policy: Callable, default_policy: Callable, backup: Callable,
                 time_limit: float = 1.0, max_iter: int = int(1e10)):
        super().__init__(tree_policy=tree_policy, default_policy=default_policy, backup=backup, time_limit=time_limit,
                         max_iter=max_iter)

        self._pipe = pipe

    def uct_search_with_pipes(self) -> None:
        while True:
            best_action = self.uct_search(self._pipe.recv())
            self._pipe.send(best_action)
