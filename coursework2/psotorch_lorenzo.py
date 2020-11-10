import numpy as np

import torch
from torch.optim import Optimizer


class PSO(Optimizer):
    """Pytorch implementation of PSO algorithm
    """

    def __init__(self, features, labels, model, loss, inertia, a1, a2, dim, population_size, time_steps, search_range):
        self.features = features
        self.labels = labels
        self.model = model
        self.loss = loss
        self.inertia = inertia
        self.a1 = a1
        self.a2 = a2
        self.dim = dim
        self.population_size = population_size
        self.time_steps = time_steps

        self.positions = np.random.uniform(low=-search_range, high=search_range,
                                           size=(self.population_size, self.dim))
        self.velocities = np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, self.dim))

        self.best_swarm_position = np.random.uniform(low=-500, high=500, size=self.dim)
        self.best_swarm_fitness = 1e30

        self.best_particle_positions = self.positions
        self.best_particle_fitnesses = np.array([1e30] * self.population_size)

    def step(self, closure=None):
        a1r1 = np.multiply(self.a1, np.random.uniform(low=0, high=1, size=(self.population_size, self.dim)))
        a2r2 = np.multiply(self.a2, np.random.uniform(low=0, high=1, size=(self.population_size, self.dim)))

        best_particle_dif = np.subtract(self.best_particle_positions, self.positions)
        best_swarm_dif = np.subtract(self.best_swarm_position, self.positions)

        self.positions += self.inertia \
                             * self.velocities \
                             + np.multiply(a1r1, best_particle_dif) \
                             + np.multiply(a2r2, best_swarm_dif)

        if np.any(self.positions.T @ self.positions > 1.0e+18):
                raise SystemExit('Most likely divergent: Decrease parameter values')

        with torch.no_grad():
            for i in range(self.population_size):
                self.update_particle_position(self.positions[i])

                if torch.cuda.is_available():
                    particle_predictions = self.model(self.features.cuda())
                    particle_fitness = self.loss(particle_predictions, self.labels.type(torch.FloatTensor).cuda().reshape(132,1))
                else:
                    particle_predictions = self.model(self.features)
                    particle_fitness = self.loss(particle_predictions, self.labels.type(torch.FloatTensor).reshape(132,1))
                
                print("before update",i,particle_fitness,self.best_particle_fitnesses[i])
                if particle_fitness < self.best_particle_fitnesses[i]:
                    self.best_particle_fitnesses[i] = particle_fitness
                    self.best_particle_positions[i] = self.positions[i]
                if particle_fitness < self.best_swarm_fitness:
                    self.best_swarm_fitness = particle_fitness
                    self.best_swarm_position = self.positions[i]
                print("after update",i,particle_fitness,self.best_particle_fitnesses[i])
                    
            self.update_particle_position(self.best_swarm_position)


                   
    def update_particle_position(self, new_position):
        old_layer_len = 0
        for name, param in self.model.named_parameters():
            if not "weight" in name:
                continue

            new_layer_len = np.prod([i for i in param.size()])
            new_position_layer_1 = new_position[old_layer_len:new_layer_len+old_layer_len]
            old_layer_len = new_layer_len


            param.copy_(torch.tensor(new_position_layer_1).reshape(param.size()))