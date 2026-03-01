from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from ir_env import IREnv, IRState
from deep_cfr_network import IRNetwork


class WeightedReservoirDataset(Dataset):
    def __init__(self, max_size: int):
        super().__init__()
        self.reservoir: list[tuple[tuple, np.ndarray]] = []
        self.max_size = max_size
        self.total_weight = 0

    def add_sample(self, infoset: tuple, target: np.ndarray, t: int):
        self.total_weight += t
        if len(self.reservoir) < self.max_size:
            self.reservoir.append((infoset, target))
        elif np.random.rand() < t * self.max_size / self.total_weight:
            replace_index = np.random.randint(self.max_size)
            self.reservoir[replace_index] = (infoset, target)
    
    def __len__(self):
        return len(self.reservoir)

    def __getitem__(self, index: int):
        infoset, target = self.reservoir[index]
        obs = torch.tensor(infoset, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float)
        return (obs, target)
    
    def save(self, fp):
        obs = np.array([infoset for infoset, _ in self.reservoir], dtype=np.int8)
        target = np.array([target for _, target in self.reservoir], dtype=np.float32)
        meta = np.array([self.max_size, self.total_weight], dtype=np.int64)
        np.savez(fp, obs=obs, target=target, metadata=meta)

    @staticmethod
    def try_from_file(fp, max_size: int):
        dataset = WeightedReservoirDataset(max_size)
        try:
            data = np.load(fp)
            reservoir = [(tuple(obs), target) for obs, target in zip(data['obs'], data['target'])]
            loaded_max_size, total_weight = data['metadata']
            data.close()
            assert loaded_max_size == max_size
            dataset.reservoir = reservoir
            dataset.total_weight = total_weight
        except FileNotFoundError:
            print("No pre-computed dataset found")
        finally:
            return dataset


class DeepMCCFR(object):
    """
    Class for implementing a Monte-Carlo counterfactual risk minimization algorithm
    """
    RESERVOIR_SIZE = 1_000_000
    N_LAYERS = 5
    HIDDEN_SIZE = 128
    EMBEDDING_SIZE = 24

    def __init__(self, env: IREnv):
        self.env = env
        self.node_touched_count = 0
        self.iteration = 0
        # Training data
        self.q_memory = [
            WeightedReservoirDataset.try_from_file("player_0_memory.npz", self.RESERVOIR_SIZE),
            WeightedReservoirDataset.try_from_file("player_1_memory.npz", self.RESERVOIR_SIZE),
        ]
        self.policy_memory = WeightedReservoirDataset(self.RESERVOIR_SIZE)
        # Networks
        self.q_networks: list[IRNetwork] = None

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

    def train_q(self, player: int):
        q_network = self.q_networks[player]
        q_network.train()
        memory = self.q_memory[player]
        dl = DataLoader(memory, batch_size=2_000, shuffle=True, pin_memory=True)
        EPOCHS = 40_000 // len(dl)

        optim = torch.optim.AdamW(q_network.parameters(), lr=1e-3, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

        _prog = tqdm(total=EPOCHS * len(dl), desc="Loss:", leave=False)
        losses = []
        ema_loss = 0
        for _ in range(EPOCHS):
            for batch in dl:
                obs, target = batch
                pred = q_network(obs)
                loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
                # Update parameters
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                optim.step()
                # Update progress
                ema_loss = 0.99 * ema_loss + 0.01 * loss.item()
                losses.append(ema_loss)
                _prog.set_description(f"Loss: {loss.item(): >7.2f}", refresh=False)
                _prog.update()
            self.sanity_check(player)
            # scheduler.step()
        q_network.eval()
        tqdm.write(f"Average loss: {ema_loss: >7.2f}")
        plt.figure(figsize=(12, 6))
        plt.plot(losses)
        plt.savefig(f"train_loss_{player}.png", dpi=300)

    def train(self, iterations: int):
        values = np.zeros(2)
        self.iteration = 1
        # Initial reservoir filling
        if min(len(self.q_memory[0]), len(self.q_memory[1])) < self.RESERVOIR_SIZE:
            _prog = tqdm(total=self.RESERVOIR_SIZE, desc="Random self-play")
            while min(len(self.q_memory[0]), len(self.q_memory[1])) < self.RESERVOIR_SIZE:
                for player in [0, 1]:
                    self.env.reset()
                    values[player] = self.external_sampling_cfr(self.env, player)
                    tqdm.write(f"Random sampling     : nodes touched = {self.node_touched_count}")
                    # tqdm.write(f"                          payoffs = {values[0]: >5.2f}  vs  {values[1]: >5.2f}")
                _prog.update(min(len(self.q_memory[0]), len(self.q_memory[1])) - _prog.n)
            _prog.close()
            del _prog
            self.q_memory[0].save("player_0_memory.npz")
            self.q_memory[1].save("player_1_memory.npz")
        else:
            print("Initializing from pre-computed memory")
        # Initial Q networks
        self.q_networks = [
            IRNetwork(5, 64, 16),
            IRNetwork(5, 64, 16),
        ]
        for player in [0, 1]:
            self.train_q(player)

        for it in tqdm(range(iterations), desc="Value directed self-play"):
            self.iteration += 1
            for player in [0, 1]:
                self.env.reset()
                values[player] = self.external_sampling_cfr(self.env, player)
            if it % 1 == 0:
                tqdm.write(f"iteration {it: >8}: nodes touched = {self.node_touched_count}")
                # tqdm.write(f"                          payoffs = {values[0]: >5.2f}  vs  {values[1]: >5.2f}")
            # Re-train Q networks
            self.q_networks = [
                IRNetwork(self.N_LAYERS, self.HIDDEN_SIZE, self.EMBEDDING_SIZE),
                IRNetwork(self.N_LAYERS, self.HIDDEN_SIZE, self.EMBEDDING_SIZE),
            ]
            for player in [0, 1]:
                self.train_q(player)

    def current_strategy(self, player: int, infoset: tuple, valid_actions: np.ndarray) -> np.ndarray:
        if self.q_networks is not None:
            with torch.no_grad():
                q_network = self.q_networks[player]
                output = q_network(torch.tensor(infoset, dtype=torch.long)[None])[0]
                values = output.cpu().numpy()
        else:
            # No trained policy yet
            return valid_actions / valid_actions.sum()
        positive_regret = np.maximum(values, 0) * valid_actions
        total = positive_regret.sum()
        if total > 0:
            return positive_regret / total
        # Uniform random
        return valid_actions / valid_actions.sum()

    def external_sampling_cfr(self, game: IREnv, player: int) -> float:
        self.node_touched_count += 1
        # Terminal
        if game.is_done():
            return game.final_points()[player]
        infoset = tuple(game.state.to_observations())
        current_player = game.whose_turn()
        if current_player != player:
            # Sample a random action
            start_state = game.state
            opposing_strategy = self.current_strategy(current_player, infoset, start_state.valid_actions())
            action = np.random.choice(15, p=opposing_strategy)
            game.step(action)
            utility = self.external_sampling_cfr(game, player)
            self.policy_memory.add_sample(infoset, opposing_strategy, self.iteration)
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
        node_value = self.current_strategy(player, infoset, start_state.valid_actions()) @ utilities
        # Sample instantaneous regret
        regret = utilities - node_value
        self.q_memory[player].add_sample(infoset, regret, self.iteration)
        return node_value
    
    def sanity_check(self, player: int):
        if player == 0:
            state = IRState(
                self.env.tiles, 1, 0,
                (0, 1, 2, 2), (0, 1, 2, 3),
                (0, 1, 2, 2), (4, 4, 4, 4),
            )
            true_reward = 7
        else:
            state = IRState(
                self.env.tiles, 0, 1,
                (0, 1, 2, 2), (4, 4, 4, 4),
                (0, 1, 2, 2, 0), (0, 1, 2, 3, 0),
            )
            true_reward = -7
        with torch.no_grad():
            q_network = self.q_networks[player]
            q_network.eval()
            output = q_network(torch.tensor(tuple(state.to_observations()), dtype=torch.long)[None])[0]
            q_network.train()
            values = output.cpu().numpy()
            positive_regret = np.maximum(values, 0) * state.valid_actions()
        action = np.argmax(positive_regret)
        value = positive_regret[action]
        self.env.state = state
        _, reward, done, _ = self.env.step(action)
        tqdm.write(f"Predicted: {value: >6.2f}    Actual: {true_reward: >2}    Location: {action % 5}")
