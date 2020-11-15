import os
from pathlib import Path

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss

from psopy.models import GenericSpiralClassifier
from psopy.optimization import PSO


def _get_accuracy(y_train, y_train_preds):
    class_classified = (y_train_preds > 0.5).float()
    accuracy = sum(y_train[i] == class_classified[i] for i in range(len(class_classified))) / y_train_preds.shape[0]
    return accuracy


class TrainingInstance:
    """
    Assumes all inputs have well defined hashing functions if wanting to use the cache.
    """

    def __init__(self, x_train, y_train, network_structure, inertia, a1, a2, population_size, search_space, epochs, seed, x_val=None, y_val=None, cache_loc=Path("/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/cache/"), device="cpu", verbose=False):

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.network_structure = list(filter(lambda a: a != 0, network_structure))
        self.model = GenericSpiralClassifier(self.network_structure).to(device)
        self.loss = BCEWithLogitsLoss().to(device)
        self.optimizer = PSO(x_train, y_train, model=self.model, loss=self.loss, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=search_space, seed=seed, dim=self.get_n_trainable_params())
        self.epochs = epochs
        self.device = device

        self.performance_cache = cache_loc / f"performances/{hash(self)}"
        self.model_cache = cache_loc / f"models/{hash(self)}"

        self.verbose = verbose

        if os.path.exists(self.performance_cache):
            self.performances = self._load_cached_performances()
            self._load_cached_model_state()
            print("Loaded from disk.")
        else:
            self.performances = pd.DataFrame(index=pd.MultiIndex.from_product([["train", "val"], ["fitness", "accuracy"]]),
                                              columns=[i for i in range(self.epochs)]
                                              )
            self._fit()

    def _fit(self):
        for i in range(self.epochs):
            y_train_preds = self.model(self.x_train)
            fitness = self.loss(y_train_preds, self.y_train)
            self.performances.loc[("train", "fitness")][i] = fitness


            accuracy = _get_accuracy(self.y_train, y_train_preds)
            self.performances.loc[("train", "accuracy")][i] = accuracy

            if self.verbose:
                print(f"Epoch {i}: Fitness = {fitness}; Training Acc = {accuracy}")

            if self.x_val is not None and self.y_val is not None:
                y_val_preds = self.model(self.x_val)
                val_fitness = self.loss(y_val_preds, self.y_val)
                val_acc = _get_accuracy(self.y_val, y_val_preds)
                self.performances.loc[("val", "fitness")][i] = val_fitness
                self.performances.loc[("val", "accuracy")][i] = val_acc

            self.optimizer.step()

        self._cache_performances()
        self._cache_model_state()

    def get_n_trainable_params(self):
        n = 0
        for name, param in self.model.named_parameters():
            if any(x in name for x in ["weight", "bias"]):
                n += param.numel()
        return n

    def get_current_performances(self):
        return self.performances.iloc[:, -1:]

    def _load_cached_performances(self):
        return pd.read_hdf(self.performance_cache, key="accuracies")
    
    def _cache_performances(self):
        self.performances.to_hdf(self.performance_cache, key="accuracies")

    def _load_cached_model_state(self):
        self.model.load_state_dict(torch.load(self.model_cache))

    def _cache_model_state(self):
        torch.save(self.model.state_dict(), self.model_cache)

    def _get_model_hash(self):
        return hash("".join(str(self.network_structure)))

    def __hash__(self):
        return hash((self.epochs, self._get_model_hash(), hash(self.optimizer), self.device))

