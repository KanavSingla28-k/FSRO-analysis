import numpy as np
from scipy.stats import levy

# Objective Function Definition

def objective_function(x):
    """Defines the objective function to be minimized."""
    return np.sum(x**2)  # Simple Sphere function


# Initialization

num_agents = 10  # Equal number of frogs and snakes
dimensions = 5
iterations = 100
search_space = (-10, 10)

# Initialize agents (frogs and snakes) with random positions
agents = np.random.uniform(search_space[0], search_space[1], (num_agents, dimensions))
best_solution = None
best_fitness = float('inf')

def crossover(parent1, parent2, beta=0.5):
    """Adaptive crossover: Beta dynamically adjusts based on fitness improvement."""
    return beta * parent1 + (1 - beta) * parent2

mylist = []

for t in range(iterations):
    
    # Fitness Evaluation
    
    fitness = np.array([objective_function(agent) for agent in agents])
    
    # Identify the best agent (solution with lowest fitness value)
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best_fitness:
        best_fitness = fitness[best_idx]
        best_solution = agents[best_idx].copy()
    
    
    # Frog Movement (Exploration & Exploitation)
    # Improvement: Adaptive Crossover (Beta adjusts with fitness difference)
    # Impact: Enhances balance between exploration and exploitation
    
    
    new_agents = np.copy(agents)
    beta = 0.5 + 0.5 * np.exp(-t / iterations)  # Beta reduces over time to favor exploitation
    for i in range(0, num_agents, 2):
        if i + 1 < num_agents:
            new_agents[i] = crossover(agents[i], agents[i+1], beta)
            new_agents[i+1] = crossover(agents[i+1], agents[i], beta)
    
    
    # Snake Predation (Capture & Inversion)
    # Improvement: Gaussian-based perturbation instead of bitwise inversion
    # Impact: More realistic predation mechanism with controlled randomness
    
    
    predation_mask = np.random.rand(num_agents, dimensions) < 0.3  # 30% chance of predation
    perturbation = np.random.normal(0, 1, (num_agents, dimensions))  # Gaussian noise
    new_agents = np.where(predation_mask, new_agents + perturbation, new_agents)
    
    
    # Fitness Improvement Calculation
    
    
    new_fitness = np.array([objective_function(agent) for agent in new_agents])
    improvement = fitness - new_fitness  # Difference between old and new fitness
    
    
    # Population Adjustment using Rank-based Selection
    # Improvement: Rank-based selection replaces softmax probability
    # Impact: Reduces premature convergence, maintains diversity
    
    
    ranks = np.argsort(np.argsort(-improvement))  # Compute ranks (higher rank = better improvement)
    selection_prob = ranks / np.sum(ranks)  # Normalize ranks to probabilities
    selected_indices = np.random.choice(range(num_agents), size=num_agents, p=selection_prob)
    agents = new_agents[selected_indices]
    
    
    # Mutation via Lévy Flight (Evolutionary Stable Strategy)
    # Improvement: Lévy Flight Mutation replaces uniform mutation
    # Impact: Introduces long jumps for better exploration in early iterations
    
    
    mutation_prob = 0.1 * (1 - t / iterations)  # Decreasing mutation rate over time
    mutation_mask = np.random.rand(num_agents, dimensions) < mutation_prob
    levy_steps = levy.rvs(size=(num_agents, dimensions)) * 0.1  # Lévy flight step size
    agents = np.where(mutation_mask, agents + levy_steps, agents)
    
    # Ensure agents remain within the search space boundaries
    agents = np.clip(agents, search_space[0], search_space[1])
    
    print(f"Iteration {t+1}: Best Fitness = {best_fitness}")
    
    mylist.append()
    


# Output Results

print("Optimization Complete.")
print("Best Solution Found:", best_solution)
print("Best Fitness:", best_fitness)
