import numpy as np
from rlearn.game import Game

class TicTacToe(Game):
    def __init__(self, n=3):
        super().__init__()
        self.n = n

    def get_init_state(self):
        '''
        state: {state, state, current_player, reward)
        reward:  1    :  if current_player wins
                -1    :  if the other player wins
                0     :  game not finished
                1e-4  :  game is tied
        '''
        b = np.zeros((self.n*self.n), dtype=int)
        return b, b, 1, 0

    def to_neural_state(self, state):
        '''
        convert to numpy, which can be fed to the neural network
        no need to convert here, as it is always in numpy
        '''
        return state

    def get_shape(self):
        '''
        dimensions of the state: (3,3)
        '''
        return (self.n, self.n)

    def get_action_size(self):
        '''
        the number of actions: each actioin is represented by the index to the board
        the last action is pass
        '''
        return self.n*self.n + 1

    def get_next_state(self, state, player, action):
        '''
        this is the critical API which controls the state transition of the game.
        if player takes action on state, return the next state
        action must be a valid move
        '''
        if action == self.n*self.n:
            return state[0], state[1], -state[2], state[3]
        next = np.copy(state[0])
        next[action] = player
        b = np.copy(next).reshape(self.n, self.n)
        if self._is_win(b, player):
            reward = 1
        elif self._is_win(b, -player):
            reward = -1
        elif len(list(zip(*np.where(next == 0))))!=0:
            reward = 0
        else:
            # draw has a very little value 
            reward = 1e-4
        return next, next, -player, reward 

    def get_valid_actions(self, state, player):
        '''
        given the current state, return the valid vector of actions
        '''
        state0 = state[0]
        valids = [0]*self.get_action_size()
        legal_moves =  list(zip(*np.where(state0 == 0)))
        if len(legal_moves)==0:
            valids[-1]=1
            return np.array(valids)
        for i in legal_moves:
            valids[i[0]]=1
        return np.array(valids)

    def get_player_agnostic_state(self, state, player):
        '''
        a state that is agonistic to the player, which is fed to the neural network for both players
        return state if player==1, else return -state if player==-1
        '''
        state0 = state[0]*player
        return state0, state0, state[2], state[3]

    def get_symmetries(self, state, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        state0 = state[0]
        b2d = state0.copy().reshape(self.n,self.n)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(b2d, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB.reshape(-1), list(newPi.ravel()) + [pi[-1]])]
        return l

    def get_game_ended(self, state, player):
        '''
        this returns the ending status of the game:
            1   : if the player wins
            -1  : if the player loses
            0   : a tie
            1e-4: game not ended
        '''
        b = np.copy(state[0]).reshape(self.n, self.n)
        if self._is_win(b, player):
            reward = 1
        elif self._is_win(b, -player):
            reward = -1
        elif len(list(zip(*np.where(b == 0))))!=0:
            reward = 0
        else:
            # draw has a very little value 
            reward = 1e-4
        return reward 

    def state_to_string(self, state):
        '''
        string representation of the state
        '''
        s = state[0]
        return ','.join([str(s[x*self.n+y]) for x in range(self.n) for y in range(self.n)])

    @staticmethod
    def display(state):
        n = 3
        state0 = state[0]
        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = state0[y*n+x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")

    def _is_win(self, state, player):
        # Check rows, columns, and diagonals
        if (np.any(np.all(state == player, axis=1)) or
            np.any(np.all(state == player, axis=0)) or
            np.all(np.diag(state) == player) or
            np.all(np.diag(np.fliplr(state)) == player)):
            return True
        return False
