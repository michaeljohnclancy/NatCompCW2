import pandas as pd

from optimization import PSO


class TrainingInstance:
    def __init__(self, instance_name, model, loss, inertia, a1, a2, population_size, search_range, epochs, seed, device="cpu"):
        self.instance_name = instance_name
        self.model = model.to(device)
        self.loss = loss.to(device)
        self.inertia = inertia
        self.a1 = a1
        self.a2 = a2
        self.population_size = population_size
        self.search_range = search_range
        self.epochs = epochs
        self.seed = seed
        self.trained = False

    def fit(self, x_train, y_train, x_val=None, y_val=None, verbose=False):
        if self.trained is True:
            raise Exception("Already trained!")

        optimizer = PSO(x_train, y_train, model=self.model, loss=self.loss, inertia=self.inertia, a1=self.a1, a2=self.a2, population_size=self.population_size, search_range=self.search_range)

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

            optimizer.step()

        self.trained = True
        self._performances = performances

    def get_performances(self):
        assert self.trained is True
        return self._performances

    def _get_accuracy(self, y_train, y_train_preds):
        class_classified = (y_train_preds > 0.5).float()
        accuracy = sum(y_train[i] == class_classified[i] for i in range(len(class_classified))) / y_train_preds.shape[0]
        return accuracy
