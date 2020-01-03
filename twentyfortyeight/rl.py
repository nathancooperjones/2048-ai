import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from twentyfortyeight.game import Game


class NeuralNetwork(nn.Module):
    """TODO."""
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 4
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.9999
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        # self.layer_1 = nn.Linear(16, 64)
        # nn.init.xavier_normal_(self.layer_1.weight)
        self.layer_2 = nn.Linear(64, 128)
        nn.init.xavier_normal_(self.layer_2.weight)
        self.layer_3 = nn.Linear(128, 256)
        nn.init.xavier_normal_(self.layer_3.weight)
        self.layer_4 = nn.Linear(256, 82)
        nn.init.xavier_normal_(self.layer_4.weight)
        self.layer_5 = nn.Linear(82, 32)
        nn.init.xavier_normal_(self.layer_5.weight)
        self.layer_6 = nn.Linear(32, self.number_of_actions)
        nn.init.xavier_normal_(self.layer_6.weight)

    def forward(self, x):
        """TODO."""
        # out = F.leaky_relu(self.layer_1(x))
        # out = F.leaky_relu(self.layer_2(out))
        out = F.leaky_relu(self.layer_2(x))
        out = F.leaky_relu(self.layer_3(out))
        out = F.leaky_relu(self.layer_4(out))
        out = F.leaky_relu(self.layer_5(out))
        out = self.layer_6(out)

        return out


def train(model, start):
    """TODO."""
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = Game()

    # initialize replay memory
    replay_memory = list()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[2] = 1  # go down because i said so
    board, reward, terminal = game_state.step(action)
    state = torch.cat((board, board, board, board)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        print(game_state.board)

        # get next state and reward
        board_1, reward, terminal = game_state.step(output)
        state_1 = torch.cat((torch.reshape(state, (4, 16))[1],
                             torch.reshape(state, (4, 16))[2],
                             torch.reshape(state, (4, 16))[3],
                             board_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        board = board_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))


def test(model):
    """TODO."""
    game_state = Game()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    board, reward, terminal = game_state.step(action)
    state = torch.cat((board, board, board, board)).unsqueeze(0)

    while True:
        # get output from the neural network
        print(game_state.board)

        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        board_1, reward, terminal = game_state.step(action)
        state_1 = torch.cat((torch.reshape(state, (4, 16))[1],
                             torch.reshape(state, (4, 16))[2],
                             torch.reshape(state, (4, 16))[3],
                             board_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1


def main(mode):
    """TODO."""
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
