import random

from IPython.display import clear_output
import numpy as np


class Game():
    """TODO."""
    def __init__(self, size=4, probability_of_spawning_4=0.1):
        # Just so I don't get confused by this naming scheme in the future:
        # [[A, 0, 0, B],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [C, 0, 0, D]]
        # A is on the upper row, left column
        # B is on the uper row, right column
        # C is on the lower row, left column
        # D is on the lower row, right column
        self.upper_row = 0
        self.left_col = 0
        self.lower_row = size - 1
        self.right_col = size - 1

        self.score = 0
        self.move_counter = 0

        self.probability_of_spawning_4 = probability_of_spawning_4

        self.move_made = None

        self.board = np.zeros((size, size)).astype(int)
        self.num_of_board_spaces = self.board.size

    def _add_random_space(self):
        """TODO."""
        list_of_zeros = list(zip(np.where(self.board == 0)[0], np.where(self.board == 0)[1]))
        if list_of_zeros:
            if random.random() < self.probability_of_spawning_4:
                self.board[random.choice(list_of_zeros)] = 4
            else:
                self.board[random.choice(list_of_zeros)] = 2
        else:
            # TODO: raise error of some sort.
            print('You lose!')

    def _recursively_move(self, row, col, direction, combine=False):
        """TODO."""
        # ensure we have proper checks and adjusted rows/cols for the specified direction
        adjusted_row_dict = {'s': row+1, 'w': row-1, 'd': row, 'a': row}
        adjusted_col_dict = {'s': col, 'w': col, 'd': col+1, 'a': col-1}
        adjusted_row = adjusted_row_dict[direction]
        adjusted_col = adjusted_col_dict[direction]

        if (direction == 's'
                and (row < self.upper_row or col < self.left_col
                     or row > self.lower_row - 1 or col > self.right_col)):
            return
        elif (direction == 'w'
              and (row - 1 < self.upper_row or col < self.left_col
                   or row > self.lower_row or col > self.right_col)
              ):
            return
        elif (direction == 'd'
              and (row < self.upper_row or col < self.left_col
                   or row > self.lower_row or col > self.right_col - 1)
              ):
            return
        elif (direction == 'a'
              and (row < self.upper_row or col - 1 < self.left_col
                   or row > self.lower_row or col > self.right_col)
              ):
            return

        # now general logic
        elif self.board[row, col] == 0:
            return
        elif combine is False and self.board[adjusted_row, adjusted_col] == 0:
            self.board[adjusted_row, adjusted_col] = self.board[row, col]
        elif combine is True and self.board[row, col] == self.board[adjusted_row, adjusted_col]:
            self.board[adjusted_row, adjusted_col] *= 2
            self.score += self.board[adjusted_row, adjusted_col]
        else:
            return

        self.board[row, col] = 0
        self.move_made = True
        if combine is False:
            self._recursively_move(adjusted_row, adjusted_col, direction, combine=False)

    def _can_we_move_this_block(self, row, col):
        """TODO."""
        if col > self.left_col:
            if self.board[row, col] == self.board[row, col - 1]:
                return True
        if col < self.right_col:
            if self.board[row, col] == self.board[row, col + 1]:
                return True
        if row > self.upper_row:
            if self.board[row, col] == self.board[row - 1, col]:
                return True
        if row < self.lower_row:
            if self.board[row, col] == self.board[row + 1, col]:
                return True
        else:
            return False

    def _transform_player_direction_input(self):
        """TODO."""
        raw_direction = input('Enter a direction [w/a/s/d or down/up/right/left]: ')
        direction = raw_direction.strip().lower()

        if direction == 'quit' or direction == 'q' or direction == 'exit':
            raise KeyboardInterrupt('Goodbye!')
        elif direction == 'right':
            direction = 'd'
        elif direction == 'left':
            direction = 'a'
        elif direction == 'up':
            direction = 'w'
        elif direction == 'down':
            direction = 's'
        if direction in ['w', 'a', 's', 'd']:
            return direction

        print(f'"{raw_direction}" is not valid. Try again or type "quit".')
        return self._transform_player_direction_input()

    def _pretty_print(self, mat, colors=True):
        """TODO."""
        print('\n'.join([''.join(
            [self._pretty_print_color_helper(item, colors) for item in row]) for row in mat])
        )

    def _pretty_print_color_helper(self, item, colors):
        if not colors:
            return '{:5}'.format(item)

        if item == 0:
            return '\033[97m {:5}\033[00m'.format(item)
        elif item == 2:
            return '\033[31m {:5}\033[00m'.format(item)
        elif item == 4:
            return '\033[91m {:5}\033[00m'.format(item)
        elif item == 8:
            return '\033[33m {:5}\033[00m'.format(item)
        elif item == 16:
            return '\033[34m {:5}\033[00m'.format(item)
        elif item == 32:
            return '\033[35m {:5}\033[00m'.format(item)
        elif item == 64:
            return '\033[36m {:5}\033[00m'.format(item)
        elif item == 128:
            return '\033[32m {:5}\033[00m'.format(item)
        elif item == 256:
            return '\033[92m {:5}\033[00m'.format(item)
        elif item == 512:
            return '\033[93m {:5}\033[00m'.format(item)
        elif item == 1024:
            return '\033[94m {:5}\033[00m'.format(item)
        elif item == 2048:
            return '\033[95m {:5}\033[00m'.format(item)
        elif item == 4096:
            return '\033[96m {:5}\033[00m'.format(item)
        else:
            return '\033[98m {:5}\033[00m'.format(item)

    def _check_if_there_is_a_move_left(self):
        """TODO."""
        # are there any 0s left?
        if len(self.board.nonzero()[0]) < self.num_of_board_spaces:
            return True
        # if not, can we move any blocks over potentially in the coming moves?
        for row in range(self.upper_row, self.lower_row + 1):
            for col in range(self.left_col, self.right_col + 1):
                if self._can_we_move_this_block(row, col):
                    return True
        return False

    def shift(self, direction):
        """TODO."""
        if self.move_made is None or self.move_made is True:
            self.move_made = False

        if direction == 's':
            # down shift everything twice, combine only once
            for iteration in range(2):
                for row_start in range(self.lower_row - 1, self.upper_row - 1, -1):
                    for row in range(row_start, self.upper_row - 1, -1):
                        for col in range(self.left_col, self.right_col + 1):
                            self._recursively_move(row, col, direction, combine=False)
                # only combine once
                if iteration == 0:
                    for row in range(self.lower_row - 1, self.upper_row - 1, -1):
                        for col in range(self.left_col, self.right_col + 1):
                            self._recursively_move(row, col, direction, combine=True)
        elif direction == 'd':
            # right shift everything twice, combine only once
            for iteration in range(2):
                for col_start in range(self.right_col - 1, self.left_col - 1, -1):
                    for col in range(col_start, self.left_col - 1, -1):
                        for row in range(self.upper_row, self.lower_row + 1):
                            self._recursively_move(row, col, direction, combine=False)
                # only combine once
                if iteration == 0:
                    for col in range(self.right_col - 1, self.left_col - 1, -1):
                        for row in range(self.upper_row, self.lower_row + 1):
                            self._recursively_move(row, col, direction, combine=True)
        elif direction == 'w':
            # up shift everything twice, combine only once
            for iteration in range(2):
                for row_start in range(self.upper_row + 1, self.lower_row + 1):
                    for row in range(row_start, self.lower_row + 1):
                        for col in range(self.left_col, self.right_col + 1):
                            self._recursively_move(row, col, direction, combine=False)
                # only combine once
                if iteration == 0:
                    for row in range(self.upper_row + 1, self.lower_row + 1):
                        for col in range(self.left_col, self.right_col + 1):
                            self._recursively_move(row, col, direction, combine=True)
        elif direction == 'a':
            # left shift everything twice, combine only once
            for iteration in range(2):
                for col_start in range(self.left_col + 1, self.right_col + 1):
                    for col in range(col_start, self.right_col + 1):
                        for row in range(self.upper_row, self.lower_row + 1):
                            self._recursively_move(row, col, direction, combine=False)
                # only combine once
                if iteration == 0:
                    for col in range(self.left_col + 1, self.right_col + 1):
                        for row in range(self.upper_row, self.lower_row + 1):
                            self._recursively_move(row, col, direction, combine=True)
        else:
            # TODO: raise error of some sort.
            print('ERROR')

        self.move_counter += 1

    def game_state(self):
        """TODO."""
        return self.board.flatten()

    def reset(self):
        """TODO."""
        self.score = 0
        self.move_counter = 0
        self.move_made = None
        self.board = np.zeros((self.lower_row + 1, self.right_col + 1)).astype(int)

        # add in some random spaces.
        for _ in range(2):
            self._add_random_space()

    def play_interactive(self, colors=True, debug=False):
        """TODO."""
        try:
            # Start by setting up the self.board.
            self.reset()

            while True:
                if debug is False:
                    clear_output()

                if self.move_made is False:
                    print("Move not made! Try again.")
                    self.move_counter -= 1

                print(f'Score: {int(self.score)}, Move Counter: {self.move_counter}')
                print('-------------------------')
                print()

                if debug is False:
                    self._pretty_print(self.board, colors)
                else:
                    print(self.board)
                print()

                # Check to make sure there is a move left we can make.
                if self._check_if_there_is_a_move_left() is False:
                    print(f'You Lose! Final self.score: {self.score}')
                    return

                # Now ask the user for input, then send that to shift.
                direction = self._transform_player_direction_input()
                self.shift(direction)
                if self.move_made:
                    # Add another random piece to the self.board.
                    self._add_random_space()

        except Exception as e:
            print(f'Uh-oh! Something went wrong! This is not good.',
                  'Here is some specifics on what went so poorly: ')
            print(e.message, e.args)
