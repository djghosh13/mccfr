import argparse
import numpy as np
import torch
from policy import CategoricalPolicyValue
from mcts import MCTS
from ir_env import IREnv, IRState
import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)


class config_ir:
    def __init__(self, seed: int = 1):
        self.env_name = "Interregnum_round1"
        self.record = False
        seed_str = "seed=" + str(seed)
        # output config
        self.output_path = "results/{}-{}/".format(
            self.env_name, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.iterations = 1000  # number of batches trained on
        self.games = 5  # number of steps used to compute each policy update
        self.simulations = 400
        self.max_replay_buffer = 20000
        self.steps_per_iteration = 5
        self.batch_size = 1000
        self.learning_rate = 3e-3

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 128
        self.embedding_size = 16
        self.max_value = 15

def predict_games(env: IREnv, policy: CategoricalPolicyValue):
    while True:
        try:
            tiles = env.tiles
            player_missing = int(input("Missing (0) Military (1) Economy (2) Religion: "))
            player_cards = []
            player_locations = []
            enemy_locations = []
            state = IRState(tiles, player_missing, None)
            for _ in range(5):
                action, value = policy.best_action(state.to_observations(), state.valid_actions())
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

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = config_ir(args.seed)
    env = IREnv(("throne room", "market", "shrine", "arena", "battleground"))

    # train model
    model = MCTS(env, config, args.seed)
    model.train()
    torch.save(model.predictor, "model.pt")

    with torch.no_grad():
        predict_games(env, model.predictor)
