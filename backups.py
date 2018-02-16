from mcts_utils import StateNode


def monte_carlo(state: StateNode, reward: float) -> None:
    while state is not None:
        state.visits += 1
        state.value += reward
        state = state.parent
