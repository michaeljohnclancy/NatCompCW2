import numpy as np
import torch
from torch import nn

from psopy.models import BaselineSpiralClassifier, LinearInputsSpiralClassifier
from psopy.preprocess import load_tensors, phi
from psopy.plotting import plot_performances

from psopy.training import TrainingInstance

seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

baseline_model = BaselineSpiralClassifier()

loss = nn.BCEWithLogitsLoss()

epochs = 10000
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/two_spirals.dat')

baseline_instance = TrainingInstance(instance_name="Baseline", model=baseline_model, loss=loss, epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=search_range, seed=seed)
initial_hyperparameter_best = TrainingInstance(instance_name="From shallow hyperparameter search", model=baseline_model, loss=loss, epochs=epochs, inertia=0.7, a1=2.8, a2=1.2, population_size=population_size, search_range=1, seed=seed)

baseline_instance.fit(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val)
initial_hyperparameter_best.fit(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val)

plot_performances(
    training_instances=[baseline_instance, initial_hyperparameter_best],
    plot_title="Hyperparameter optimization starting point",
    fitness_name="Binary Cross Entropy loss",
    save_location="/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/figures/shallowhyperparameterbest.pdf")
