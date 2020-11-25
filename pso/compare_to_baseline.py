import numpy as np
import torch

from plotting import plot_performances
from preprocess import load_tensors, phi

from training import TrainingInstance

seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

epochs = 10000
inertia = 0.1
a1 = 2.9
a2 = 1.1
population_size = 30
search_range = 10

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/two_spirals.dat')

training_instance = TrainingInstance(
        x_train=phi(x_train), y_train=y_train, x_val=phi(x_val), y_val=y_val, 
        network_structure=[6,8,1],
        inertia=inertia, a1=a1, a2=a2, population_size=population_size, search_space=search_range, 
        seed=seed, epochs=epochs
        )

plot_performances(
    training_instances=[training_instance],
    plot_title="Optimal hyperparameters",
    fitness_name="Binary Cross Entropy loss",
    save_location="/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/figures/comparisontolinearinputs.pdf")
