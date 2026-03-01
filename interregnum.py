from dataclasses import dataclass
import numpy as np

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


TILES = {
    name: (value, np.array(["a" in ttype, "b" in ttype, "c" in ttype], dtype=int))
    for name, value, ttype in [
        (     "wasteland", 1, "abc"),

        (         "arena", 2, "a"),
        (          "shop", 2, "b"),
        (        "garden", 2, "c"),

        (  "battleground", 3, "a"),
        (        "bazaar", 3, "b"),
        (        "meadow", 3, "c"),
        (    "guard post", 3, "ab"),
        (        "valley", 3, "ac"),
        (        "shrine", 3, "bc"),

        (       "warzone", 4, "a"),
        (        "market", 4, "b"),
        (      "mountain", 4, "c"),
        ("mercenary camp", 4, "ab"),
        (         "tower", 4, "ac"),
        (        "temple", 4, "bc"),

        (    "city state", 5, "ab"),
        (   "throne room", 5, "ac"),
        (     "cathedral", 5, "bc"),
    ]
}

STRENGTH = {
    "a": np.array([1, 0, 0]),
    "b": np.array([0, 1, 0]),
    "c": np.array([0, 0, 1]),
}


@dataclass(frozen=True)
class State:
    tiles: tuple[str]
    player_missing: str
    enemy_missing: str
    player_cards: tuple[str] = ()
    player_locations: tuple[int] = ()
    enemy_cards: tuple[str] = ()
    enemy_locations: tuple[int] = ()

@dataclass(frozen=True)
class Infoset:
    tiles: tuple[str]
    player_missing: str
    player_cards: tuple[str] = ()
    player_locations: tuple[int] = ()
    enemy_locations: tuple[int] = ()

def is_done(state: State) -> bool:
    return len(state.player_locations) == len(state.enemy_locations) == 5

def whose_turn(state: State) -> int:
    return int(len(state.player_locations) < len(state.enemy_locations))

def to_infoset(state: State) -> Infoset:
    return Infoset(
        state.tiles, state.player_missing,
        state.player_cards, state.player_locations,
        state.enemy_locations
    )

def state_points(state: State) -> tuple[int, int]:
    assert is_done(state)
    # Count usable strength
    types_per_tile = np.array([TILES[tile][1] for tile in state.tiles])
    player_strength = np.zeros((len(state.tiles), 3), dtype=int)
    for card, loc in zip(state.player_cards, state.player_locations):
        player_strength[loc] += STRENGTH[card]
    player_strength = (player_strength * types_per_tile).max(axis=1)
    enemy_strength = np.zeros((len(state.tiles), 3), dtype=int)
    for card, loc in zip(state.enemy_cards, state.enemy_locations):
        enemy_strength[loc] += STRENGTH[card]
    enemy_strength = (enemy_strength * types_per_tile).max(axis=1)
    # Total points
    points_per_tile = np.array([TILES[tile][0] for tile in state.tiles])
    player_points = points_per_tile @ (player_strength > enemy_strength)
    enemy_points = points_per_tile @ (enemy_strength > player_strength)
    return player_points, enemy_points

def valid_actions(state: State) -> list[tuple[str, int]]:
    valid = {"a": 2, "b": 2, "c": 2}
    valid[state.player_missing] -= 1
    for card in state.player_cards:
        valid[card] -= 1
    return [
        (card, loc)
        for card in "abc"
        for loc in range(len(state.tiles))
        if valid[card] > 0
    ]

def take_action(state: State, action: tuple[str, int]) -> State:
    card, loc = action
    new_player_cards = state.player_cards + (card,)
    new_player_locations = state.player_locations + (loc,)
    return State(
        state.tiles, state.player_missing, state.enemy_missing,
        state.enemy_cards, state.enemy_locations,
        new_player_cards, new_player_locations,
    )

def ideal_value(my_points: int, their_points: int) -> float:
    return my_points - their_points


# def compute_value(state: State) -> float:
#     if is_done(state):
#         return ideal_value(*state_points(state))
#     for action in valid_actions(state):
#         next_state = take_action(state, action)


def traverse(state: State, player: int, agent_0, agent_1, memory_v: list, memory_pi: list, t: int) -> float:
    infoset = to_infoset(state)
    if is_done(state):
        return state_points(state)[player]
    
    elif whose_turn(state) == player:
        all_actions = valid_actions(state)
        # Compute policy
        pred_values = {}
        for action in all_actions:
            pred_values[action] = (agent_0 if player == 0 else agent_1)(infoset, action)
        total_regret = sum(max(v, 0) for v in pred_values.values())
        probs = {}
        if total_regret > 0:
            for action in all_actions:
                probs[action] = pred_values[action] / total_regret
        else:
            for action in all_actions:
                probs[action] = 1 / len(all_actions)
        # Compute real values
        values = {}
        for action in all_actions:
            values[action] = traverse(take_action(state, action), player, agent_0, agent_1, memory_v, memory_pi, t)
        exp_value = sum(probs[action] * values[action] for action in all_actions)
        advantages: dict[tuple[str, int], float] = {}
        for action in valid_actions(state):
            advantages[action] = values[action] - exp_value
        # Insert into memory_v
        memory_v.append((infoset, t, advantages))
        return exp_value # TODO: Idk

    else:
        all_actions = valid_actions(state)
        # Compute policy
        pred_values = {}
        for action in all_actions:
            pred_values[action] = (agent_1 if player == 0 else agent_0)(infoset, action)
        total_regret = sum(max(v, 0) for v in pred_values.values())
        probs: dict[tuple[str, int], float] = {}
        if total_regret > 0:
            for action in all_actions:
                probs[action] = pred_values[action] / total_regret
        else:
            for action in all_actions:
                probs[action] = 1 / len(all_actions)
        # Insert into memory_pi
        memory_pi.append((infoset, t, probs))
        action_list = list(probs.keys())
        action = action_list[np.random.choice(len(probs.keys()), p=list(probs.values()))]
        return traverse(take_action(state, action), player, agent_0, agent_1, memory_v, memory_pi, t)
    

TOTAL_ITERATIONS = 10
ITERATIONS = 1000
def deep_cfr(initial_state: State):
    agent_0 = lambda *_: 0
    agent_1 = lambda *_: 0
    memory_v0 = []
    memory_v1 = []
    memory_pi = []
    for t in tqdm(range(TOTAL_ITERATIONS)):
        for player in [0, 1]:
            for k in tqdm(range(ITERATIONS)):
                traverse(
                    initial_state, player, agent_0, agent_1,
                    memory_v0 if player == 0 else memory_v1, memory_pi, t
                )
            memory_v0 = memory_v0[-100_000:]
            memory_v1 = memory_v1[-100_000:]
            memory_pi = memory_pi[-1_000_000:]
            if player == 0:
                agent_0 = train_from_scratch(memory_v0)
            else:
                agent_1 = train_from_scratch(memory_v1)
    return memory_pi


class MyDataset(Dataset):
    def __init__(self, memory_v: list[tuple[Infoset, int, dict[tuple[str, int], float]]]):
        super().__init__()
        self.memory = memory_v
    
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, index: int):
        infoset, t, values = self.memory[index]
        all_numbers = [
            "abc".index(infoset.player_missing)
        ] + [
            "abc".index(infoset.player_cards[i]) if len(infoset.player_cards) > i else 3
            for i in range(5)
        ] + [
            infoset.player_locations[i] if len(infoset.player_locations) > i else 5
            for i in range(5)
        ] + [
            infoset.enemy_locations[i] if len(infoset.enemy_locations) > i else 5
            for i in range(5)
        ]
        all_values = [
            values.get((card, loc), None)
            for card in "abc"
            for loc in range(5)
        ]
        return (
            torch.tensor(all_numbers, dtype=torch.long),
            torch.tensor(t, dtype=torch.float),
            torch.tensor(all_values, dtype=torch.float),
        )
        

def train_from_scratch(memory_v: list[tuple[Infoset, int, dict[tuple[str, int], float]]]):
    ds = MyDataset(memory_v)
    dl = DataLoader(ds, batch_size=1000, shuffle=True, num_workers=2)
    for batch in dl:
        pass
    return lambda *_: 0


class RandomPolicy:
    def distribution(self, obs: torch.LongTensor):
        with torch.no_grad():
            batch_dims = obs.shape[:-1]
            player_played = obs[..., :6]
            times_played = torch.zeros((*batch_dims, 4), dtype=torch.long)
            for index in range(6):
                times_played[player_played[..., index]] += 1
            can_play = (times_played[..., :3] < 2).long().unsqueeze(-1).tile(5).reshape(-1, 3 * 5)
            return torch.distributions.Categorical(probs=can_play)
    
    def act(self, obs: np.ndarray):
        with torch.no_grad():
            dist = self.distribution(obs)
            dist.sample().numpy()


def run_game(tiles: tuple[str, str, str, str, str], policy_1: RandomPolicy, policy_2: RandomPolicy):
    missing_1 = "abc"[np.random.randint(3)]
    missing_2 = "abc"[np.random.randint(3)]
    state = State(tiles, missing_1, missing_2)



if __name__ == "__main__":
    s0 = State(("cathedral", "city state", "market", "valley", "wasteland"), "a", "b")
    deep_cfr(s0)
