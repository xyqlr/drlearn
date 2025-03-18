import logging
from tqdm import tqdm
from collections import deque

class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, player1, player2, game, display=None, learn_from_play=False):
        """
        Input:
            player 1,2: two functions that take the board as input, return action
            game: Game object
            display: a function that takes the board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.learn_from_play = learn_from_play
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        state = self.game.get_init_state()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "start_game"):
                player.start_game()
        play_data = []
        ended = False
        while ended == 0:
            it += 1
            if verbose:
                assert self.display
                logging.debug("Turn ", str(it), "Player ", str(current_player))
                self.display(state)
            state_ag = self.game.get_player_agnostic_state(state, current_player)
            action = players[current_player + 1](state_ag)
            if self.learn_from_play:
                pi = [0] * self.game.get_action_size()
                pi[action] = 1
                sym = self.game.get_symmetries(state_ag, pi)
                for b, p in sym:
                    play_data.append([b, current_player, p])


            valids = self.game.get_valid_actions(state_ag, 1)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent of the move
            opponent = players[-current_player + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(state, action)

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            ended = self.game.get_game_ended(state, current_player)

        for player in players[0], players[2]:
            if hasattr(player, "end_game"):
                player.end_game()

        reward = ended * current_player

        if verbose:
            assert self.display
            logging.debug("Game over: Turn ", str(it), "Result ", str(reward))
            self.display(state)
            if reward == 1:
                print("The first player won!")
            elif reward == -1:
                print("The second player won!")
            else:
                print("It is a tie")
        if self.learn_from_play:
            play_data = [(x[0], x[2], ended * x[1]) for x in play_data]
        return reward, play_data

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws: games won by nobody
        """

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        train_data = deque([])
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            game_result, play_data = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            if self.learn_from_play:
                train_data += play_data
        if self.game.alternate_turn:
            self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            game_result, play_data = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
            if self.learn_from_play:
                train_data += play_data

        return one_won, two_won, draws, train_data

    def eval_game(self, start_state, player_func):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        current_player = 1
        state = start_state
        ended = False
        while ended == 0:
            action = player_func(state)

            valids = self.game.get_valid_actions(state, current_player)

            if valids[action] == 0:
                logging.error(f'Action {action} is not valid!')
                logging.debug(f'valids = {valids}')
                assert valids[action] > 0

            state = self.game.get_next_state(state, current_player, action)
            current_player = state[2]
            ended = state[3]

        return ended * current_player

    def eval_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws: games won by nobody
        """

        one_won = 0
        two_won = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.eval_games"):
            start_state = self.game.get_init_state()
            result1 = self.eval_game(start_state, self.player1)
            result2 = self.eval_game(start_state, self.player2)
            if result1 > result2:
                one_won += 1
            elif result1 < result2:
                two_won += 1
            else:
                draws += 1

        return one_won, two_won, draws