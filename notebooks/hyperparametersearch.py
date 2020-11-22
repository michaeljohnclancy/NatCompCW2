import itertools
import json
import numpy as np
import torch

import os


from preprocess import load_tensors, phi

from training import TrainingInstance

from progress.bar import Bar

os.environ['PYTHONHASHSEED'] = '0'
seed = 12345324
np.random.seed(seed)
torch.random.manual_seed(seed)

epochs = 2000

population_size = 30

x_train, y_train, x_val, y_val, x_test, y_test = load_tensors('/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/two_spirals.dat')

inertia_range = np.round(np.arange(0.1,1.0,0.1), decimals=2)
a1_range = np.round(np.arange(0, 5.1, 0.1), decimals=2)
a2_range = a1_range[::-1]

a1a2_pairs = []
for pair in itertools.product(a1_range, a2_range):
    if sum(pair) == 4.0:
        a1a2_pairs.append(pair)

search_size_range = np.round(np.logspace(-1, 3, num=5), decimals=2)

param_combinations = list(itertools.product(inertia_range, a1a2_pairs, search_size_range))
print(f"Number of hyperparameter combinations to test: {len(param_combinations)}")

best_validation_loss = 1e10

bar = Bar('Hyperparameter set', max=len(param_combinations))

for i, params in enumerate(param_combinations):
    inertia = params[0]
    a1 = params[1][0]
    a2 = params[1][1]
    search_size = params[2]
    print(f"Iteration {i}, {len(param_combinations) - i} to go")
    print(f"Params: Inertia={inertia}; a1={a1}; a2={a2}; search_size={search_size};")

    training_instance = TrainingInstance(
        x_train=phi(x_train), y_train=y_train, x_val=phi(x_val),
        y_val=y_val, network_structure=[6,8,1], inertia=inertia,
        a1=a1, a2=a2, population_size=population_size,
        search_space=search_size, seed=seed, epochs=epochs)

    validation_loss = training_instance.get_current_performances().loc[("train", "fitness")].iloc[0]
    validation_accuracy = training_instance.get_current_performances().loc[("train", "accuracy")].iloc[0]

    print(f" Validation Loss={validation_loss}; Validation Accuracy={validation_accuracy}")
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_training_instance = training_instance

    bar.next()

with open("/home/mclancy/Documents/notes/edinburgh/year4/naturalcomputing/coursework/data/hyperparamsearch/bestparams.json", 'w') as fp:
    json.dump(str(best_training_instance.pso_params), fp)