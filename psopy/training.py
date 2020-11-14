import os
from pathlib import Path

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss

from psopy.models import GenericSpiralClassifier


class TrainingInstance:
    """
    Assumes all inputs have well defined hashing functions if wanting to use the cache.
    """

    def __init__(self, network_layers, optimizer, epochs, cache_loc=Path("/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/cache/"), device="cpu",):

        self.network_layers = network_layers
        self.model = GenericSpiralClassifier(self.network_layers).to(device)
        self.loss = BCEWithLogitsLoss().to(device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.trained = False

        self.performance_cache = cache_loc / f"performances/{hash(self)}"
        self.model_cache = cache_loc / f"models/{hash(self)}"

        if os.path.exists(self.performance_cache):
            self._performances = self._load_cached_performances()
            self._load_cached_model_state()
            self.trained = True
            print("Loaded from disk.")


    def fit(self, x_train, y_train, x_val=None, y_val=None, verbose=False):
        if self.trained is True:
            raise Exception("Already trained")

        performances = pd.DataFrame(index=pd.MultiIndex.from_product([["train", "val"], ["fitness", "accuracy"]]),
                                    columns=[i for i in range(self.epochs)]
                                    )

        for i in range(self.epochs):
            y_train_preds = self.model(x_train)
            fitness = self.loss(y_train_preds, y_train)
            performances.loc[("train", "fitness")][i] = fitness

            accuracy = self._get_accuracy(y_train, y_train_preds)
            performances.loc[("train", "accuracy")][i] = accuracy

            if verbose:
                print(f"Epoch {i}: Fitness = {fitness}; Training Acc = {accuracy}")

            if x_val is not None and y_val is not None:
                y_val_preds = self.model(x_val)
                val_fitness = self.loss(y_val_preds, y_val)
                val_acc = self._get_accuracy(y_val, y_val_preds)
                performances.loc[("val", "fitness")][i] = val_fitness
                performances.loc[("val", "accuracy")][i] = val_acc

            self.optimizer.step()

        self.trained = True
        self._performances = performances
        self._cache_performances()
        self._cache_model_state()

    @property
    def performances(self):
        assert self.trained is True
        return self._performances

    def _get_accuracy(self, y_train, y_train_preds):
        class_classified = (y_train_preds > 0.5).float()
        accuracy = sum(y_train[i] == class_classified[i] for i in range(len(class_classified))) / y_train_preds.shape[0]
        return accuracy

    def _load_cached_performances(self):
        return pd.read_hdf(self.performance_cache, key="accuracies")
    
    def _cache_performances(self):
        self._performances.to_hdf(self.performance_cache, key="accuracies")

    def _load_cached_model_state(self):
        self.model.load_state_dict(torch.load(self.model_cache))

    def _cache_model_state(self):
        torch.save(self.model.state_dict(), self.model_cache)

    def _get_model_hash(self):
        return hash("".join(str(self.network_layers)))

    def __hash__(self):
        return hash((self.epochs, self._get_model_hash(), hash(self.optimizer), self.device))
