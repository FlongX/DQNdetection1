import sys
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%m%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class EarlyStopping:
    def __init__(self, patience=100, patience_lr=10, patience_t=2):
        self.patience = patience
        self.patience_lr = patience_lr
        self.patience_t = patience_t
        self.counter = 0
        self.counter_lr = 0
        self.counter_t = 0
        self.best_dice = 0

    def EStop(self, val_dice):
        decay = 1
        early_stop = False
        save_model = False
        if val_dice > self.best_dice:
            self.best_dice = val_dice
            self.counter = 0
            self.counter_lr = 0

            save_model = True
        else:
            self.counter += 1
            self.counter_lr += 1
            if self.counter_lr >= self.patience_lr:
                decay = 0.5
                self.counter_lr = 0
            if self.counter >= self.patience:
                early_stop = True
        return decay, early_stop, save_model
