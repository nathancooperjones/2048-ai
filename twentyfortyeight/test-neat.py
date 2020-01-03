import os
import pickle

import neat
from twentyfortyeight.game import Game


def run_game(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = Game()

    while not game.game_over:
        game._pretty_print(game.board)
        print()

        output = net.activate(game.game_state())

        _, _, _ = game.step(output)

    game._pretty_print(game.board)
    print()

    print(game.score)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    genome_file = 'winner-feedforward'
    genome = pickle.load(open(genome_file, 'rb'))

    run_game(genome, config)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path)
