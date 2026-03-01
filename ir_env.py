from typing import Any, NamedTuple
from dataclasses import dataclass

import numpy as np

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

STRENGTH = [
    np.array([3, 1, 2]), # Military
    np.array([2, 3, 1]), # Economy
    np.array([1, 2, 3]), # Religion
]


@dataclass(frozen=True)
class IRState:
    tiles: tuple[str]
    player_missing: int
    enemy_missing: int
    player_cards: tuple[int] = ()
    player_locations: tuple[int] = ()
    enemy_cards: tuple[int] = ()
    enemy_locations: tuple[int] = ()

    def to_observations(self) -> np.ndarray:
        # Observations: my missing card, my played cards, my locations, enemy locations
        all_obs = [self.player_missing] + [
            self.player_cards[i] if len(self.player_cards) > i else 3
            for i in range(5)
        ] + [
            self.player_locations[i] if len(self.player_locations) > i else 5
            for i in range(5)
        ] + [
            self.enemy_locations[i] if len(self.enemy_locations) > i else 5
            for i in range(5)
        ]
        return np.array(all_obs, dtype=int)
    
    def valid_actions(self) -> np.ndarray:
        remaining = [2, 2, 2]
        remaining[self.player_missing] -= 1
        for card in self.player_cards:
            remaining[card] -= 1
        is_valid = np.zeros((3, 5), dtype=bool)
        for card in range(3):
            if remaining[card] > 0:
                is_valid[card] = True
        return is_valid.flatten()


@dataclass(frozen=True)
class IRAction:
    card: int
    location: int

class Result(NamedTuple):
    state: IRState
    reward: tuple[float, float]
    done: bool
    info: Any = None


class IREnv(object):
    OBS_DIM = 1 + 5 + 5 + 5 # my missing card, my played cards, my locations, enemy locations
    ACT_DIM = 3 * 5 # card types x locations

    def __init__(self, tiles: tuple[str]):
        self.tiles = tiles
        self.state = None
        self.rng = np.random.default_rng()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
    
    def reset(self) -> IRState:
        self.state = IRState(
            self.tiles,
            self.rng.integers(3),
            self.rng.integers(3),
        )
        return self.state

    def step(self, action: int) -> Result:
        assert self.state is not None
        action = IRAction(action // 5, action % 5)
        new_player_cards = self.state.player_cards + (action.card,)
        new_player_locations = self.state.player_locations + (action.location,)
        self.state = IRState(
            self.state.tiles, self.state.player_missing, self.state.enemy_missing,
            self.state.enemy_cards, self.state.enemy_locations,
            new_player_cards, new_player_locations,
        )
        if self.is_done():
            return Result(self.state, self.final_points(), True)
        return Result(self.state, (0, 0), False)
    
    def is_done(self) -> bool:
        return len(self.state.player_locations) == len(self.state.enemy_locations) == 5
    
    def whose_turn(self) -> int:
        return int(len(self.state.player_locations) < len(self.state.enemy_locations))
    
    def action_list(self) -> list[int]:
        return list(np.arange(15)[self.state.valid_actions()])

    def final_points(self) -> tuple[int, int]:
        # Count usable strength
        types_per_tile = np.array([TILES[tile][1] for tile in self.state.tiles])
        player_strength = np.zeros((len(self.state.tiles), 3), dtype=int)
        for card, loc in zip(self.state.player_cards, self.state.player_locations):
            player_strength[loc] += STRENGTH[card]
        player_strength = (player_strength * types_per_tile).max(axis=1)
        enemy_strength = np.zeros((len(self.state.tiles), 3), dtype=int)
        for card, loc in zip(self.state.enemy_cards, self.state.enemy_locations):
            enemy_strength[loc] += STRENGTH[card]
        enemy_strength = (enemy_strength * types_per_tile).max(axis=1)
        # Total points
        points_per_tile = np.array([TILES[tile][0] for tile in self.state.tiles])
        player_points = points_per_tile @ (player_strength > enemy_strength)
        enemy_points = points_per_tile @ (enemy_strength > player_strength)
        return player_points - enemy_points, enemy_points - player_points
        return player_points, enemy_points
