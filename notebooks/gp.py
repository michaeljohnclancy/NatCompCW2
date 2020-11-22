import numpy as np

import torch

import torch.nn as nn

from preprocess import load_tensors, phi
from models import GenericSpiralClassifier

from training import TrainingInstance

class GP:
    
    def __init__(self, x_train, y_train, D, N, T, p_c, p_m, seed, max_hidden_units=10, dev="cpu", inertia=0.7, a1=1.5, a2=1.8, population_size=30, search_range=1, x_val=None, y_val=None, phi=lambda x:x):

        self.x_train = phi(x_train)
        self.y_train = y_train
        self.x_val = phi(x_val) if x_val is not None else None
        self.y_val = y_val

        self.loss = nn.BCEWithLogitsLoss().to(dev)
        self.dev = dev

        self.D = D  # Dimension of the search space
        self.N = N  # Size of the population of solutions, usually much larger
        self.T = T  # Number of generations. Often needs to be larger than this.
        self.p_c = p_c  # Crossover probability
        self.p_m = p_m  # Mutation probbability
        self.min_hidden_units = 0
        self.max_hidden_units = max_hidden_units

        self.elitism = 0  # A binary switch for whether elitism is used (1) or not (0)
        self.population = self.initalize_population()

        self.inertia = inertia
        self.a1 = a1
        self.a2 = a2
        self.population_size = population_size
        self.search_range = search_range
        self.seed = seed

    def initalize_population(self):
        """
        Initalize the original population.
        Note: the first layer can only have features from 1-6 and the output needs to be 1.

        THIS NEEDS TO BE GENERALIZED FOR ANY SIZE OF LAYERS. CURRENTLY ONLY WORKDS FOR 3.
        """

        start = np.random.binomial(n=1, p=0.5, size=(self.N, 6))
       
        member_genes = []
        for member_index in range(self.N):
            member_hidden_units = []
            for layer_index in range(self.D - 2):
                number_hidden_units = np.random.randint(self.max_hidden_units)
                activation_function_choice = np.random.choice(['A', 'B', 'C'])
                member_hidden_units.append([number_hidden_units, activation_function_choice])
                
            member_genes.append([start[member_index], member_hidden_units, 1])

        init_pop = member_genes
        return init_pop

    def fitness_func(self, num_epochs):
        """
        Evaluate each of the members of the population, the members of the population are of the form [a,b,1]
        We need to train the NN for this structure and then evaluate the fitness after they have been trained.

        Return: - the population sorted form best to worst fitness (smallest to largest values)
                - the fitness of the population
        """


        fitness_list = np.zeros(self.N)
        i = 0

        
        for member in self.population:
            print(f"Currently training: {member}.")
            
            network_structure = []
            for i in range(len(member)):
                if i == 0:
                    network_structure.append(np.sum(member[i]))
                elif i == 1:
                    for x in member[i]:
                        network_structure.append(x[0])
                else:
                    network_structure.append(1)
            
            nonlinearities = []
            for i in range(len(member[1])):
                if member[1][i][0] != 0:
                    nonlinearities.append(member[1][i][1])
                
            training_instance = TrainingInstance(x_train=get_feature_subset(self.x_train, member[0]),
                                                 y_train=self.y_train, x_val=get_feature_subset(self.x_val, member[0]),
                                                 y_val=self.y_val, network_structure=network_structure,
                                                 nonlinearity_keys=nonlinearities,
                                                 inertia=self.inertia, a1=self.a1,a2=self.a2, 
                                                 population_size=self.population_size, search_space=self.search_range, 
                                                 seed=self.seed, epochs=num_epochs)

            fitness = training_instance.get_current_performances().loc[('val', "fitness")]
            fitness_list[i] = fitness

            print(f"{member} with fitness {fitness}.")
            fitness_indices = np.argsort(fitness_list)
            sorted_pop = [self.population[i] for i in fitness_indices]
            i += 1

        fitness_list = fitness_list[fitness_indices]

        return sorted_pop, 1 / fitness_list

    def roulette_wheel_selection(self, sorted_pop, fitness_list):

        intermediate_pop = np.zeros((self.N, self.D))
        select_from = np.arange(self.N)
        total_fit = np.sum(fitness_list)

        if total_fit == 0:
            total_fit = 1
            relative_fitness = fitness_list + 1 / self.N
        else:
            relative_fitness = fitness_list / total_fit

        mating_population = np.random.choice(select_from, self.N, p=relative_fitness)

        for member in range(len(mating_population)):
            intermediate_pop[member] = sorted_pop[mating_population[member]]

        return intermediate_pop

    def new_generation(self, intermediate_pop):

        new_pop = np.zeros((self.N, self.D))
        parent_list = np.arange(self.N)
        pairings = np.random.choice(parent_list, (2, int(self.N / 2)), replace=False)
        for x in range(np.int(self.N / 2)):
            parent1 = pairings[0][x]
            parent2 = pairings[1][x]
            new_pop[x], new_pop[(self.N - 1) - x] = self.crossover(intermediate_pop[parent1], intermediate_pop[parent2])
        self.mutate(new_pop)

        return new_pop

    def crossover(self, parent1, parent2):

        c_point = np.random.randint(0, self.D - 1)  # Crossover point since last element will always be 1
        child1 = np.zeros(self.D)
        child2 = np.zeros(self.D)
        for chromosome in range(c_point):
            child1[chromosome] = parent1[chromosome]
            child2[chromosome] = parent2[chromosome]
        for chromosome in range(self.D - c_point):
            child1[c_point + chromosome] = parent2[c_point + chromosome]
            child2[c_point + chromosome] = parent1[c_point + chromosome]

        return child1, child2

    def mutate(self, population):

        for member_index in range(len(population)):
            for chromosome_type_index in range(self.D - 1):
                if np.random.rand() < self.p_m:
                    if chromosome_type_index == 0:
                        population[member_index][chromosome_type_index] = [not x if np.random.rand() < self.p_m else x for x in population[member_index][chromosome_type_index]]
                    else:
                        population[member_index][chromosome_type_index][0] = np.random.randint(self.max_hidden_units)
                        population[member_index][chromosome_type_index][1] = np.rand.choice(["A","B","C"])

        return population

    def run(self, num_epochs):

        cur_gen = self.population

        for t in range(self.T):
            # Evaluate fitness for current generation
            sorted_pop, fitness_list = self.fitness_func(num_epochs)
            # Selection process
            intermediate_pop = self.roulette_wheel_selection(sorted_pop, fitness_list)
            # Update for the next generation, here is where we crossover and mutation
            self.population = self.new_generation(intermediate_pop)
            print(t), print(self.population)
            
        sorted_pop, fitness_list = self.fitness_func(num_epochs)
        return sorted_pop[0]

def get_feature_subset(data, phi_bool_map):
    return data[np.nonzero(phi_bool_map)]