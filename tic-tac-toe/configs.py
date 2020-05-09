class Configs():
    def __init__(self):
        self.BOARD_COLS = 3
        self.BOARD_ROWS = 3
        self.POLICIES_DIR = './policies'
        self.lr = 0.2
        self.decay_gamma = 0.9
        self.exp_rate = 0.3
        self.training_epoch = 50000