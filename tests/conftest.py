import logging
import pytest
from drlearn import nnargs
from drlearn.tictactoe import TicTacToe, TicTacToeModel
from drlearn.blackjack import BlackJack, BlackJackModel

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def setup_tictactoe_game():
    game = TicTacToe()
    model = TicTacToeModel(game, nnargs)
    return game, model

@pytest.fixture
def setup_blackjack_game():
    game = BlackJack()
    model = BlackJackModel(game, nnargs)
    return game, model

