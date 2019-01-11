import numpy as np
import torch

# modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if target value dosen't improve after a given patience."""
    def __init__(self, patience=5, verbose=True, for_max=True):
        """
        Args:
            patience (int): How long to wait after last time target value improved.
                            Default: 5
            verbose (bool): If True, prints a message for each counter. 
                            Default: True
            for_max (bool): If True, larger value denotes better performance
                            Default: True
        """
        self.patience = patience
        self.verbose = verbose
        self.for_max=for_max
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        

    def __call__(self, value, logger):

        score = value if self.for_max else -value

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0 