import numpy as np
import matplotlib.pyplot as plt

# Sphere function
def sphere_function(x):
    return np.sum(np.square(x))

# Initialize population
def initialize_population(pop_size, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

# Fitness function
def fitness_function(population):
    return np.apply_along_axis(sphere_function, 1, population)

# Crossover (blend crossover for continuous values)
def crossover(parent1, parent2, alpha=0.5):
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

# Mutation (Gaussian mutation)
def mutate(individual, mutation_rate=0.1, sigma=0.1):
    if np.random.rand() < mutation_rate:
        mutation = np.random.normal(0, sigma, size=individual.shape)
        return individual + mutation
    return individual

# Evolutionary Process
def fsro_optimization(pop_size=50, dim=10, bounds=(-5.12, 5.12), generations=100):
    population = initialize_population(pop_size, dim, bounds)
    best_fitness = []
    
    for gen in range(generations):
        fitness = fitness_function(population)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]  # Sort population by fitness
        best_fitness.append(fitness[sorted_indices[0]])
        
        new_population = []
        for i in range(0, pop_size, 2):
            if i+1 < pop_size:
                child1, child2 = crossover(population[i], population[i+1])
                child1, child2 = mutate(child1), mutate(child2)
                new_population.extend([child1, child2])
        
        population = np.array(new_population)
    
    return best_fitness

# Run FSRO Optimization
fsro_results = fsro_optimization()

# Compare with Standard Genetic Algorithm (GA)
def ga_optimization(pop_size=50, dim=10, bounds=(-5.12, 5.12), generations=100):
    population = initialize_population(pop_size, dim, bounds)
    best_fitness = []
    
    for gen in range(generations):
        fitness = fitness_function(population)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        best_fitness.append(fitness[sorted_indices[0]])
        
        new_population = population[:pop_size//2]
        while len(new_population) < pop_size:
            p1, p2 = np.random.choice(pop_size//2, 2, replace=False)
            child1, child2 = crossover(population[p1], population[p2])
            child1, child2 = mutate(child1), mutate(child2)
            np.append(new_population,[child1, child2])
            
        
        population = np.array(new_population)
    
    return best_fitness

ga_results = ga_optimization()

# Plot Results
plt.plot(fsro_results, label='FSRO')
plt.plot(ga_results, label='GA')
plt.xlabel('Generations')
plt.ylabel('Best Fitness (Sphere Function)')
plt.title('FSRO vs. GA Optimization Performance')
plt.legend()
plt.show()