import numpy as np
import pickle

BOARD_COLS = 3
BOARD_ROWS = 3

class State():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        self.board = np.zeros((BOARD_COLS, BOARD_ROWS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def getAvailablePositions(self):
        positions = []
        for i in range(BOARD_COLS):
            for j in range(BOARD_ROWS):
                if self.board[i][j] == 0:
                    positions.append((i, j))

        return positions

    def winner(self):
        """
        Values on board: Player 1: 1 || Player 2: -1
        To check if there is a winner/draw
        """

        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1

            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        # col
        for j in range(BOARD_COLS):
            if sum(self.board[:, j]) == 3:
                self.isEnd = True
                return 1
            
            if sum(self.board[:, j]) == -3:
                self.isEnd = True
                return -1

        # diagonal
        diag_sum1 = sum(self.board[i, i] for i in range(BOARD_COLS))
        diag_sum2 = sum(self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS))
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
        self.board = np.zeros((BOARD_COLS, BOARD_ROWS))
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

    def play(self, rounds=100):
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
    
    # play with human
    def play2(self):
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
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
        # self.p1.savePolicy()
        # self.p2.savePolicy()


class Player():
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.states_val = {}
        self.lr = 0.2
        self.decay_gamma = 0.9
        self.exp_rate = exp_rate

    def getHash(self, board):
        return str(board.reshape(BOARD_COLS, BOARD_ROWS))

    def chooseAction(self, positions, board, playerSymbol):
        # choose randomly
        if np.random.normal(0,1,1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]

        # exploitation using greedy method
        else:
            max_value = -999
            for p in positions:
                tmp_board = board.copy()
                tmp_board[p] = playerSymbol
                tmp_hashBoard = self.getHash(tmp_board)

                value = 0 if self.states_val.get(tmp_hashBoard) is None else self.states_val.get(tmp_hashBoard)

                if value > max_value:
                    max_value = value
                    action = p

        return action

    def addStates(self, board_hash):
        self.states.append(board_hash)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_val.get(st) is None:
                self.states_val[st] = 0

            self.states_val[st] += self.lr * (self.decay_gamma * reward - self.states_val[st])
            reward = self.states_val[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_val, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        print('initializing')
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    ## play between bots
    # p1 = Player("player 1")
    # p2 = Player("player 2")

    # st = State(p1, p2)
    # st.play(50000)

    ## play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_player 1")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()