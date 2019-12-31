import numpy as np
from twentyfortyeight.game import Game

GAME = Game()


def test_check_if_there_is_a_move_left():
    # Test that we can correctly detect when legal moves remain.
    GAME.board = np.reshape(np.arange(0, 16), (4, 4))
    assert GAME._check_if_there_is_a_move_left() is True
    GAME.board = np.reshape(np.arange(1, 17), (4, 4))
    assert GAME._check_if_there_is_a_move_left() is False
    GAME.board = np.reshape(np.ones(16), (4, 4))
    assert GAME._check_if_there_is_a_move_left() is True
    GAME.reset()
    assert GAME._check_if_there_is_a_move_left() is True


def test_shift_down_1():
    # Test shift down
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [4, 0, 0, 0],
                         [4, 2, 2, 2]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('s')

    assert np.array_equal(GAME.board, expected)


def test_shift_down_2():
    # Test shift down special consideration #1
    expected = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [4, 0, 0, 0],
                         [4, 2, 2, 2]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [0, 0, 0, 0],
                           [2, 0, 0, 0],
                           [4, 0, 0, 0]])
    GAME.shift('s')

    assert np.array_equal(GAME.board, expected)


def test_shift_down_3():
    # Test shift down special consideration #2
    expected = np.array([[0, 0, 0, 0],
                         [4, 0, 0, 0],
                         [4, 0, 0, 0],
                         [8, 2, 2, 2]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [4, 0, 0, 0],
                           [8, 0, 0, 0]])
    GAME.shift('s')

    assert np.array_equal(GAME.board, expected)


def test_shift_up_1():
    # Test shift up
    expected = np.array([[4, 2, 2, 2],
                         [4, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('w')

    assert np.array_equal(GAME.board, expected)


def test_shift_up_2():
    # Test shift up special consideration #1
    expected = np.array([[4, 2, 2, 2],
                         [4, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [0, 0, 0, 0],
                           [2, 0, 0, 0],
                           [4, 0, 0, 0]])
    GAME.shift('w')

    assert np.array_equal(GAME.board, expected)


def test_shift_up_3():
    # Test shift up special consideration #2
    expected = np.array([[4, 2, 2, 2],
                         [4, 0, 0, 0],
                         [8, 0, 0, 0],
                         [0, 0, 0, 0]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [4, 0, 0, 0],
                           [8, 0, 0, 0]])
    GAME.shift('w')

    assert np.array_equal(GAME.board, expected)


def test_shift_right_1():
    # Test shift right
    expected = np.array([[0, 0, 4, 4],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('d')

    assert np.array_equal(GAME.board, expected)


def test_shift_right_2():
    # Test shift right special consideration #1
    expected = np.array([[0, 0, 4, 4],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2]])

    GAME.board = np.array([[2, 0, 2, 4],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('d')

    assert np.array_equal(GAME.board, expected)


def test_shift_right_3():
    # Test shift right special consideration #2
    expected = np.array([[0, 4, 4, 8],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2],
                         [0, 0, 0, 2]])

    GAME.board = np.array([[2, 2, 4, 8],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('d')

    assert np.array_equal(GAME.board, expected)


def test_shift_left_1():
    # Test shift left
    expected = np.array([[4, 4, 0, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0]])

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('a')

    assert np.array_equal(GAME.board, expected)


def test_shift_left_2():
    # Test shift left special consideration #1
    expected = np.array([[4, 4, 0, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0]])

    GAME.board = np.array([[2, 0, 2, 4],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('a')

    assert np.array_equal(GAME.board, expected)


def test_shift_left_3():
    # Test shift down special consideration #2
    expected = np.array([[4, 4, 8, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0],
                         [2, 0, 0, 0]])

    GAME.board = np.array([[2, 2, 4, 8],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('a')

    assert np.array_equal(GAME.board, expected)


def test_variables_are_set():
    # Test `reset()`, `MOVE_COUNTER`, and `SCORE` values
    GAME.reset()
    assert GAME.move_counter == 0
    assert GAME.score == 0
    assert GAME.move_made is None
    assert len(GAME.board.nonzero()[0]) == 2

    GAME.board = np.array([[2, 2, 2, 2],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0],
                           [2, 0, 0, 0]])
    GAME.shift('a')
    GAME.shift('d')
    GAME.shift('w')
    GAME.shift('s')
    assert GAME.move_counter == 4
    assert GAME.score == 20
    assert GAME.move_made is True

    GAME.reset()
    assert GAME.move_counter == 0
    assert GAME.score == 0
    assert GAME.move_made is None
    assert len(GAME.board.nonzero()[0]) == 2


def test_moves_dont_count():
    GAME.reset()
    GAME.board = np.array([[2, 0, 0, 2],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
    GAME.shift('w')
    assert GAME.move_made is False
    GAME.shift('w')
    assert GAME.move_made is False
    GAME.shift('s')
    assert GAME.move_made is True
