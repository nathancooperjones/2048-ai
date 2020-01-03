import os
import pickle

import neat
import numpy as np
from twentyfortyeight.game import Game
import twentyfortyeight.visualize as visualize


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = Game()

        while not game.game_over:
            output = net.activate(game.game_state())

            _, _, _ = game.step(output)

        # game._pretty_print(game.board)
        # print()

        genome.fitness = float(game.score) + (0.5 * np.max(game.board))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    # Run for up to 1000 generations.
    winner = p.run(eval_genomes, 1000)
    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=True, view=True, filename='feedforward-fitness.svg')
    visualize.plot_species(stats, view=True, filename='feedforward-speciation.svg')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path)
