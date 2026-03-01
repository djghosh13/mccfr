import os
from tqdm import tqdm
import numpy as np

from ir_env import IREnv, IRState

class CFRNode:
    def __init__(self, valid_actions: np.ndarray):
        self.valid_actions = valid_actions.copy()
        self.regret_sum = np.zeros(15)
        self.strategy = self.uniform()
        self._strategy_sum = np.zeros(15)
        self._average_strategy = np.zeros(15)
        self._already_calculated: bool = False
        self._need_to_update_strategy: bool = False
    
    def average_strategy(self) -> np.ndarray:
        if not self._already_calculated:
            self._calculate_average_strategy()
        return self._average_strategy
    
    def strategy_sum(self, strategy: np.ndarray, weight: float):
        self._strategy_sum += strategy * weight
        self._already_calculated = False
    
    def set_regret_sum(self, values: np.ndarray):
        self.regret_sum = values.copy()
        self._need_to_update_strategy = True

    def update_strategy(self):
        if not self._need_to_update_strategy:
            return
        self.strategy = np.maximum(self.regret_sum, 0)
        total = self.strategy.sum()
        if total > 0:
            self.strategy /= total
        else:
            self.strategy = self.uniform()
        self._need_to_update_strategy = False # Need?
    
    def _calculate_average_strategy(self):
        if self._already_calculated:
            return
        self.average_strategy[...] = 0
        total = self._strategy_sum.sum()
        if total > 0:
            self._average_strategy = self._strategy_sum / total
        else:
            self._average_strategy = self.uniform()
        self._already_calculated = True
    
    def uniform(self) -> np.ndarray:
        return np.full(15, 1 / self.valid_actions.sum()) * self.valid_actions


class MCCFR(object):
    """
    Class for implementing a Monte-Carlo counterfactual risk minimization algorithm
    """
    def __init__(self, env: IREnv):
        self.env = env
        self.node_map: dict[tuple, CFRNode] = {}
        self.node_touched_count = 0

    def calculate_payoff(self, game: IREnv) -> np.ndarray:
        # Terminal
        if game.is_done():
            return np.array(game.final_points())
        # Chance node (only initial)
        # TODO
        # Player node
        player = game.whose_turn()
        node_utility = np.array([0, 0], dtype=float)
        start_state = game.state
        for action in range(15):
            game.state = start_state
            game.step(action)
            node_utility += self.strategy_prob(action, player) * self.calculate_payoff(game)
        game.state = start_state
        return node_utility
    
    def calculate_exploitability(self, game: IREnv) -> float:
        infosets: dict[tuple, list[tuple[IRState, float]]] = {}
        pass

    def create_infosets(self, infosets: dict[tuple, list[tuple[IRState, float]]],
                        game: IREnv, player: int, prob: float):
        # Terminal
        if game.is_done():
            return
        # Chance node (only initial)
        # TODO
        # Player node
        if game.whose_turn() == player:
            infoset = tuple(game.state.to_observations())
            infosets.setdefault(infoset, []).append((game.state, prob))
            # Traverse actions
            start_state = game.state
            for action in game.action_list():
                game.state = start_state
                game.step(action)
                self.create_infosets(infosets, game, player, prob)
            game.state = start_state
        else:
            # Traverse actions
            start_state = game.state
            for action in game.action_list():
                game.state = start_state
                game.step(action)
                self.create_infosets(infosets, game, player, prob * self.strategy_prob(action, player))
            game.state = start_state

    def train(self, iterations: int):
        values = np.zeros(2)
        for it in tqdm(range(iterations)):
            for player in [0, 1]:
                # Continue if not update-player
                self.env.reset()
                values[player] = self.external_sampling_cfr(self.env, player)
            if it % 10 == 0:
                tqdm.write(f"iteration {it: >8}: nodes touched = {self.node_touched_count}")
                tqdm.write(f"                     num infosets = {len(self.node_map)}")
                tqdm.write(f"                          payoffs = {values[0]: >5.2f}  vs  {values[1]: >5.2f}")

    def external_sampling_cfr(self, game: IREnv, player: int) -> float:
        self.node_touched_count += 1
        # Terminal
        if game.is_done():
            return game.final_points()[player]
        infoset = tuple(game.state.to_observations())
        current_player = game.whose_turn()
        node = self.node_map.setdefault(infoset, CFRNode(game.state.valid_actions()))
        node.update_strategy()
        if current_player != player:
            # Sample a random action
            start_state = game.state
            action = np.random.choice(15, p=node.strategy)
            game.step(action)
            utility = self.external_sampling_cfr(game, player)
            node.strategy_sum(node.strategy, 1.0)
            game.state = start_state
            return utility
        # Iterate over all actions
        utilities = np.zeros(15)
        start_state = game.state
        for action in game.action_list():
            game.state = start_state
            game.step(action)
            utilities[action] = self.external_sampling_cfr(game, player)
        game.state = start_state
        node_value = node.strategy @ utilities
        # Compute CFR
        node.set_regret_sum(node.regret_sum + utilities - node_value)
        return node_value
