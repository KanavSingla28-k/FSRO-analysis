import numpy as np

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

def crossover(parent1, parent2):
    """Performs crossover between two agents to generate new solutions."""
    alpha = np.random.rand()
    return alpha * parent1 + (1 - alpha) * parent2

for t in range(iterations):
    # Fitness Evaluation
    
    fitness = np.array([objective_function(agent) for agent in agents])
    
    # Identify the best agent (solution with lowest fitness value)
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < best_fitness:
        best_fitness = fitness[best_idx]
        best_solution = agents[best_idx].copy()
    
    # Frog Movement (Exploration & Exploitation)
    
    new_agents = np.copy(agents)
    for i in range(0, num_agents, 2):
        if i + 1 < num_agents:
            new_agents[i] = crossover(agents[i], agents[i+1])
            new_agents[i+1] = crossover(agents[i+1], agents[i])
    
    # Snake Predation (Capture & Inversion)
    
    predation_points = np.random.choice([0, 1], size=(num_agents, dimensions))
    new_agents = np.where(predation_points == 1, 1 - new_agents, new_agents)  # Inversion at 0 and 1
    
    # Fitness Improvement Calculation
    
    new_fitness = np.array([objective_function(agent) for agent in new_agents])
    improvement = fitness - new_fitness  # Difference between old and new fitness
    
    # Population Adjustment using Replicator Dynamics
    
    probabilities = np.exp(improvement) / np.sum(np.exp(improvement))  # Compute selection probabilities
    selected_indices = np.random.choice(range(num_agents), size=num_agents, p=probabilities)
    agents = new_agents[selected_indices]  # Select new agents based on probabilities
    
    # Mutation via Evolutionary Stable Strategy
    
    mutation_prob = 0.1  # Probability of mutation
    mutation_mask = np.random.rand(num_agents, dimensions) < mutation_prob  # Mask for mutation selection
    mutation_values = np.random.uniform(-1, 1, (num_agents, dimensions))  # Random perturbation values
    agents = np.where(mutation_mask, agents + mutation_values, agents)  # Apply mutation
    
    # Ensure agents remain within the search space boundaries
    agents = np.clip(agents, search_space[0], search_space[1])
    
    print(f"Iteration {t+1}: Best Fitness = {best_fitness}")

# Output Results

print("Optimization Complete.")
print("Best Solution Found:", best_solution)
print("Best Fitness:", best_fitness)
