from dataclasses import dataclass, field
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from network_utils import np2torch, IRPolicyValueNetwork
from policy import CategoricalPolicyValue
from ir_env import IREnv, IRState


@dataclass
class MCTSNode:
    prior: float
    to_play: int # 1 or -1
    children: dict[int, 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0
    state: IRState = None

    def value(self):
        return self.value_sum / self.visit_count
    
    def ucb_score(self, child: 'MCTSNode'):
        weighted_prior = child.prior * np.sqrt(self.visit_count) / (child.visit_count + 1)
        value_score = -child.value() if child.visit_count else 0
        return value_score + weighted_prior
    
    def posteriors(self) -> dict[int, float]:
        return {
            action: child.visit_count / self.visit_count
            for action, child in self.children.items()
        }
    
    def select_child(self) -> tuple[int, 'MCTSNode']:
        return max(self.children.items(), key=lambda action_child: self.ucb_score(action_child[1]))


class MCTS(object):
    """
    Class for implementing a Monte-Carlo tree search algorithm
    """

    def __init__(self, env: IREnv, config, seed: int):
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed
        self.env = env
        self.env.seed(self.seed)

        self.observation_dim = self.env.OBS_DIM
        self.action_dim = self.env.ACT_DIM

        self.lr = self.config.learning_rate
        self.replay_buffer = []

        self.init_policy()

    def init_policy(self):
        #######################################################
        #########   YOUR CODE HERE - 8-12 lines.   ############
        network = IRPolicyValueNetwork(self.config.n_layers, self.config.layer_size, self.config.embedding_size, self.config.max_value)
        self.predictor = CategoricalPolicyValue(network)
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=self.lr, weight_decay=1e-3)
        #######################################################
        #########          END YOUR CODE.          ############

    def add_samples(self, observations: list, probs: list, values: list):
        self.replay_buffer.extend(
            dict(observation=o, probs=p, value=v)
            for o, p, v in zip(observations, probs, values)
        )
        while len(self.replay_buffer) > self.config.max_replay_buffer:
            self.replay_buffer.pop(0)

    def run_simulation_turns(self, root: MCTSNode, env: IREnv, num_simulations: int):
        # Expand root node
        self.expand_node(root)
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            # Select
            while node.children:
                action, node = node.select_child()
                search_path.append(node)
            # Simulate
            parent = search_path[-2]
            env.state = parent.state
            next_state, reward, done, _ = env.step(action)
            node.state = next_state
            if not done:
                # Expand leaf node
                value = self.expand_node(node)
            else:
                value = reward[0]
            # Backup
            self.backup_path(search_path, value)
    
    def run_games(self, env: IREnv, to_play: int, num_games: int):
        EPSILON = 1e-3
        ZERO_THRESHOLD = 1e-9
        for _ in tqdm(range(num_games), leave=False, desc="Games"):
            state = env.reset()
            all_observations = []
            all_probs = []
            all_values = []
            for _ in range(10):
                root = MCTSNode(0, to_play, state=state)
                self.run_simulation_turns(root, env, self.config.simulations)
                # Collect sample
                observations = state.to_observations()
                posteriors = root.posteriors()
                probs = np.array([
                    posteriors[action] if action in posteriors else 0
                    for action in range(15)
                ])
                value = root.value()
                all_observations.append(observations)
                all_probs.append(probs)
                all_values.append(value)
                # Take random action
                explore_probs = np.zeros(15)
                explore_probs[probs > ZERO_THRESHOLD] = (
                    (1 - EPSILON) * probs[probs > ZERO_THRESHOLD] + EPSILON / (probs > ZERO_THRESHOLD).sum()
                )
                action = np.random.choice(15, p=explore_probs)
                env.state = state
                state, reward, done, _ = env.step(action)
                if done:
                    # value = float(reward[1])
                    # for _ in range(len(observations)):
                    #     all_values.insert(0, value)
                    #     value *= -1
                    self.add_samples(all_observations, all_probs, all_values)
                    break
                

    def expand_node(self, node: MCTSNode):
        ZERO_THRESHOLD = 1e-9
        with torch.no_grad():
            distribution, value = self.predictor.action_distribution(
                np2torch(node.state.to_observations()[None]),
                node.state.valid_actions()[None]
            )
            probs = distribution.probs.numpy()[0]
            value = value.squeeze().numpy()
        if node.children:
            return value
        for action, prob in enumerate(probs):
            if prob > ZERO_THRESHOLD:
                node.children[action] = MCTSNode(prob, -node.to_play)
        return value
    
    def backup_path(self, search_path: list[MCTSNode], value: float):
        for node in search_path[::-1]:
            node.visit_count += 1
            node.value_sum += value
            value *= -1

    def update_policy(self):
        batch_ids = np.random.choice(len(self.replay_buffer), self.config.batch_size, replace=True)

        observations = np2torch(np.array([self.replay_buffer[idx]['observation'] for idx in batch_ids]))
        search_probs = np2torch(np.array([self.replay_buffer[idx]['probs'] for idx in batch_ids]))
        search_values = np2torch(np.array([self.replay_buffer[idx]['value'] for idx in batch_ids]))
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        distributions, values = self.predictor.action_distribution(observations)
        pred_logits = distributions.logits
        pred_value = values.squeeze(1)
        loss: torch.Tensor = F.mse_loss(pred_value, search_values) + F.cross_entropy(pred_logits, search_probs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #######################################################
        #########          END YOUR CODE.          ############

    def train(self):
        for t in tqdm(range(self.config.iterations), desc="Iterations"):
            self.run_games(self.env, 1, self.config.games)

            for _ in range(self.config.steps_per_iteration):
                self.update_policy()

            if t % 10 == 0:
                # tqdm.write(f"Values: {np.abs(search_values).mean():.2f} +/- {np.abs(search_values).std():2f}")
                self.sanity_check()
                tqdm.write("")
    
    def sanity_check(self):
        with torch.no_grad():
            state = IRState(
                self.env.tiles, 0, 1,
                (0, 1, 2, 2), (0, 1, 2, 3),
                (0, 1, 2, 2, 0), (4, 4, 4, 4, 4),
            )
            action, value = self.predictor.best_action(state.to_observations(), state.valid_actions())
            self.env.state = state
            _, reward, done, _ = self.env.step(action)
            tqdm.write(f"Predicted: {value: >6.2f}    Actual: {reward[1]: >2}    Location: {action % 5}")

            state = IRState(
                self.env.tiles, 0, 1,
                (0, 1, 2, 2), (4, 4, 4, 4),
                (0, 1, 2, 2, 0), (0, 1, 2, 3, 0),
            )
            action, value = self.predictor.best_action(state.to_observations(), state.valid_actions())
            self.env.state = state
            _, reward, done, _ = self.env.step(action)
            tqdm.write(f"Predicted: {value: >6.2f}    Actual: {reward[1]: >2}    Location: {action % 5}")

            # state = IRState(
            #     self.env.tiles, 0, 1,
            #     (0, 1, 2, 2), (4, 4, 4, 4),
            #     (0, 1, 2, 2, 0), (0, 1, 2, 3, 0),
            # )
            # root = MCTSNode(0, -1, state=state)
            # self.run_simulation_turns(root, self.env, 100)
            # posteriors = root.posteriors()
            # action = max(posteriors.keys(), key=lambda a: posteriors[a])
            # loc_probs = [sum(posteriors.get(unit * 5 + loc, 0) for unit in range(3)) for loc in range(5)]
            # value = root.value()
            # self.env.state = state
            # _, reward, done, _ = self.env.step(action)
            # tqdm.write(f"Predicted: {value: >6.2f}    Actual: {reward[1]: >2}    Probs: {loc_probs}")

            # state = IRState(
            #     self.env.tiles, 1, 0,
            #     (0, 1, 2, 2), (0, 1, 2, 3),
            #     (0, 1, 2, 2), (4, 4, 4, 4),
            # )
            # root = MCTSNode(0, 1, state=state)
            # self.run_simulation_turns(root, self.env, 100)
            # posteriors = root.posteriors()
            # action = max(posteriors.keys(), key=lambda a: posteriors[a])
            # loc_probs = [sum(posteriors.get(unit * 5 + loc, 0) for unit in range(3)) for loc in range(5)]
            # value = root.value()
            # tqdm.write(f"Predicted: {value: >6.2f}    Actual: {7: >2}    Probs: {loc_probs}")
