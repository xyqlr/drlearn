import logging
import pytest
from rlearn import TicTacToe, HumanTicTacToePlayer, TicTacToeModel, nnargs

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def setup_tictactoe_game():
    game = TicTacToe()
    model = TicTacToeModel(game, nnargs)
    human_player = HumanTicTacToePlayer(game)
    return game, model, human_player

