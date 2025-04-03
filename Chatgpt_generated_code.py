import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize FSRO Parameters
POP_SIZE = 20  # Total population (frogs + snakes)
MAX_GEN = 50  # Number of generations
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2

# Generate Synthetic Data for Feature Selection
X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Population (Binary Representation for Feature Selection)
def initialize_population(size, features):
    return np.random.choice([0, 1], size=(size, features))

# Fitness Function: Evaluate Classification Accuracy
def fitness(individual):
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) == 0:
        return 0  # Avoid empty feature set
    
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train[:, selected_features], y_train)
    predictions = clf.predict(X_test[:, selected_features])
    return accuracy_score(y_test, predictions)

# Two-Point Crossover (Snake Strategy)
def two_point_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    
    return child1, child2

# Uniform Crossover (Frog Strategy)
def uniform_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    
    mask = np.random.randint(0, 2, size=len(parent1))
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    
    return child1, child2

# Mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# Evolutionary Process
def FSRO():
    population = initialize_population(POP_SIZE, X.shape[1])
    best_solution = None
    best_fitness = 0
    
    for generation in range(MAX_GEN):
        fitness_values = np.array([fitness(ind) for ind in population])
        
        sorted_indices = np.argsort(fitness_values)[::-1]
        population = population[sorted_indices]  # Sort by fitness
        
        if fitness_values[0] > best_fitness:
            best_fitness = fitness_values[0]
            best_solution = population[0]
        
        new_population = []
        for i in range(0, POP_SIZE, 2):
            if i+1 >= POP_SIZE:
                break
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = two_point_crossover(parent1, parent2)
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])
        
        population = np.array(new_population)
    
    return best_solution, best_fitness

# Run FSRO Algorithm
best_features, best_acc = FSRO()
selected_features = np.where(best_features == 1)[0]
print(f"Selected Features: {selected_features}")
print(f"Best Classification Accuracy: {best_acc:.4f}")
