import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience, delta, path='model.pt'):
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.counter = 0
        self.best_score = -np.inf
        self.path=path
    def __call__(self, val_acc, model):
        if val_acc > self.best_score + self.delta:
            self.best_score = val_acc
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True

                
