import argparse
import numpy as np
from mccfr import MCCFR, CFRNode
from ir_env import IREnv, IRState
import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)


def predict_games(env: IREnv, solver: MCCFR):
    while True:
        try:
            tiles = env.tiles
            player_missing = int(input("Missing (0) Military (1) Economy (2) Religion: "))
            player_cards = []
            player_locations = []
            enemy_locations = []
            state = IRState(tiles, player_missing, None)
            for _ in range(5):
                infoset = tuple(state.to_observations())
                node = solver.node_map.get(infoset, CFRNode(state.valid_actions()))
                if infoset not in solver:
                    print(f"Warning: unexplored infoset")
                action, value = policy.best_action(infoset, state.valid_actions())
                print(f"Expected value:", value)
                print(f"I play> {'MER'[action // 5]} to {tiles[action % 5]}")
                player_cards.append(action // 5)
                player_locations.append(action % 5)
                enemy_loc = tiles.index(input("Enemy played to: ").strip().lower())
                enemy_locations.append(enemy_loc)
                state = IRState(
                    tiles, player_missing, None,
                    tuple(player_cards), tuple(player_locations),
                    (), tuple(enemy_locations),
                )
        except Exception:
            print("Try again")


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    env = IREnv(("throne room", "market", "shrine", "arena", "battleground"))

    # train model
    solver = MCCFR(env)
    solver.train(100)
