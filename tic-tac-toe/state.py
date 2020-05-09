import numpy as np

from configs import Configs

class State():
    def __init__(self, p1, p2):
        self.configs = Configs()
        self.BOARD_COLS = self.configs.BOARD_COLS
        self.BOARD_ROWS = self.configs.BOARD_ROWS
        self.POLICIES_DIR = self.configs.POLICIES_DIR

        self.p1 = p1
        self.p2 = p2

        self.board = np.zeros((self.BOARD_COLS, self.BOARD_ROWS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape(self.BOARD_COLS * self.BOARD_ROWS))
        return self.boardHash

    def getAvailablePositions(self):
        positions = []
        for i in range(self.BOARD_COLS):
            for j in range(self.BOARD_ROWS):
                if self.board[i][j] == 0:
                    positions.append((i, j))

        return positions

    def winner(self):
        """
        Values on board: Player 1: 1 || Player 2: -1
        To check if there is a winner/draw
        """

        # row
        for i in range(self.BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1

            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        # col
        for j in range(self.BOARD_COLS):
            if sum(self.board[:, j]) == 3:
                self.isEnd = True
                return 1
            
            if sum(self.board[:, j]) == -3:
                self.isEnd = True
                return -1

        # diagonal
        diag_sum1 = sum(self.board[i, i] for i in range(self.BOARD_COLS))
        diag_sum2 = sum(self.board[i, self.BOARD_COLS - i - 1] for i in range(self.BOARD_COLS))
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))

        if diag_sum == 3:
            if diag_sum1 == 3 or diag_sum2 == 3:
                self.isEnd = True
                return 1

            else:
                self.isEnd = True
                return -1

        # draw
        if len(self.getAvailablePositions()) == 0:
            self.isEnd = True
            return 0

        # game continues
        self.isEnd = False
        return None

    def updateStates(self, position):
        self.board[position] = self.playerSymbol

        # Switch player
        self.playerSymbol = 1 if self.playerSymbol == -1 else -1

    def reset(self):
        self.board = np.zeros((self.BOARD_COLS, self.BOARD_ROWS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1 

    def giveReward(self):
        """
        At game end only
        """
        result = self.winner()

        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)

        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)

        else:
            # if its a draw
            self.p1.feedReward(0.1) # less reward 
            self.p2.feedReward(0.5) # to make p1 more aggressive

    # play between bots
    def playBot(self, rounds=100):
        print("Initialize training for {} epochs".format(rounds))
        for i in range(rounds):
            if i % 1000 == 0:
                print('Rounds {}'.format(i))
            
            while not self.isEnd:
                # player 1
                positions = self.getAvailablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateStates(p1_action)
                board_hash = self.getHash()
                self.p1.addStates(board_hash)

                # check if win
                winner = self.winner()

                if winner is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # player 2
                    positions = self.getAvailablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateStates(p2_action)
                    board_hash = self.getHash()
                    self.p2.addStates(board_hash)

                    winner = self.winner()

                    if winner is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

        print("Done training... Saving 2 policies to {}".format(self.POLICIES_DIR))
        self.p1.savePolicy(self.POLICIES_DIR)
        self.p2.savePolicy(self.POLICIES_DIR)

    # play with human
    def playHuman(self):
        while not self.isEnd:
            # Player 1
            positions = self.getAvailablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateStates(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.getAvailablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateStates(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break
    
    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, self.BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')