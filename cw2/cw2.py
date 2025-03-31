import numpy as np
import random
import matplotlib.pyplot as plt

def shift_r(x):
    return np.concatenate((np.zeros_like(x[..., :, -1:]), x[..., :, :-1]), axis=-1)

def shift_l(x):
    return np.concatenate((x[..., :, 1:], np.zeros_like(x[..., :, :1])), axis=-1)

def shift_t(x):
    return np.concatenate((np.zeros_like(x[..., -1:, :]), x[..., :-1, :]), axis=-2)

def shift_b(x):
    return np.concatenate((x[..., 1:, :], np.zeros_like(x[..., :1, :])), axis=-2)

def evaluate(x, size=20):
    assert np.shape(x)[-1] == size * size
    grid = np.asarray(np.reshape(x, (-1, size, size)), dtype=np.int_)
    points = np.minimum(
        shift_r(grid) + shift_l(grid) + shift_t(grid) + shift_b(grid),
        1 - grid
    )
    return points.reshape(np.shape(x)).sum(-1)

def initialize_population(pop_size, size):
    return np.random.randint(0, 2, (pop_size, size * size))  # Binary grid (flattened)

def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    if total_fitness == 0:
        return random.choice(population)  # Prevent division by zero
    selection_probs = [f / total_fitness for f in fitness]
    return population[np.random.choice(len(population), p=selection_probs)]

def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]  # Flip bit (binary mutation)
    return solution

def genetic_algorithm(size=20, pop_size=50, generations=100, mutation_rate=0.1):
    population = initialize_population(pop_size, size)
    
    for gen in range(generations):
        fitness = [evaluate(sol, size) for sol in population]
        new_population = []
        
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitness)
            parent2 = roulette_wheel_selection(population, fitness)
            
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]  # Maintain population size
        
        if gen % 10 == 0:
            best_fitness = max(fitness)
            print(f'Generation {gen}, Best Fitness: {best_fitness}')
    
    best_solution = max(population, key=lambda sol: evaluate(sol, size))
    print('Best Solution Fitness:', evaluate(best_solution, size))

    
    return best_solution

best = genetic_algorithm()
size = 20
# Random solution
ran = np.random.randint(0, 2, (1, size * size))
ran_grid = np.reshape(ran, (size, size))
ran_solution = evaluate(ran)
plt.imshow(ran_grid, cmap='gray')
plt.title("Random Solution = " + str(ran_solution))
plt.colorbar()
plt.show()




# Trivial solution
grid = np.zeros((20, 20), dtype=int)

# Alternate rows, starting with 1 for odd rows
grid[1::2, ::2] = 1  # Odd rows, even columns
grid[::2, 1::2] = 1  # Even rows, odd columns

# Flatten the grid into a vector
trivial = grid.flatten()
# trivial = np.tile([0, 1], 200)
trivial_grid = np.reshape(trivial, (size, size))
trivial_solution = evaluate(trivial)
plt.imshow(trivial_grid, cmap='gray')
plt.title("Trivial Solution Chessboard = " + str(trivial_solution))
plt.colorbar()
plt.show()
trivial = np.tile([0, 1], 200)
trivial_grid = np.reshape(trivial, (size, size))
trivial_solution = evaluate(trivial)
plt.imshow(trivial_grid, cmap='gray')
plt.title("Trivial Solution Zebra = " + str(trivial_solution))
plt.colorbar()
plt.show()


# Visualization
best_solution = evaluate(best)
size = 20
best_grid = np.reshape(best, (size, size))
plt.imshow(best_grid, cmap='gray')
plt.title("Best Solution = " + str(best_solution))
plt.colorbar()
plt.show()
