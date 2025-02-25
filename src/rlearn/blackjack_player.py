class HumanBlackJackPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, state):
        valid = self.game.get_valid_actions(state, 1)
        str = f"Enter 0 for hit\n" if valid[0] else f""
        str += f"Enter 1 for stand" if valid[1] else f""
        print(str)
        while True:
            a = input()
            a = int(a)
            if valid[a]:
                break
            else:
                print('Invalid')
        return a