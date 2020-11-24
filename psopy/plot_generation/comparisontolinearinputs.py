import numpy as np
import torch
from torch import nn

from psopy.modules.models import BaselineSpiralClassifier, LinearInputsSpiralClassifier
from preprocess import load_tensors, phi
from plotting import plot_performances

from training import TrainingInstance

seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

baseline_model = BaselineSpiralClassifier()
linear_inputs_model = LinearInputsSpiralClassifier()

loss = nn.BCEWithLogitsLoss()

epochs = 10000
inertia = 0.7
a1 = 1.5
a2 = 1.8
population_size = 30
search_range = 10

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/two_spirals.dat')

baseline_instance = TrainingInstance(instance_name="Baseline", model=baseline_model, loss=loss, epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=search_range, seed=seed)
linear_inputs_instance = TrainingInstance(instance_name="Linear inputs only", model=linear_inputs_model, loss=loss, epochs=epochs, inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_range=search_range, seed=seed)

baseline_instance.fit(x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val)
linear_inputs_instance.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

# plot_accuracies([training_instance], )
plot_performances(
    training_instances=[baseline_instance, linear_inputs_instance],
    plot_title="Comparison with linear input only network",
    fitness_name="Binary Cross Entropy loss",
    save_location="/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/figures/comparisontolinearinputs.pdf")
