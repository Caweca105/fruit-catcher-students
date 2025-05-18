import random
import inspect

def create_individual(individual_size):
    """
    Create a single individual represented by a real-valued vector.
    """
    return [random.uniform(-1, 1) for _ in range(individual_size)]


def generate_population(individual_size, population_size):
    """
    Generate an initial population of given size.
    """
    return [create_individual(individual_size) for _ in range(population_size)]


def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.2,
    mutation_rate=0.05
):
    """
    Evolve a population of real-valued vectors via a genetic algorithm.

    Parameters:
    - individual_size: int, number of genes per individual
    - population_size: int, number of individuals in population
    - fitness_function: callable, computes fitness from individual (and optional seed)
    - target_fitness: float, fitness threshold for stopping
    - generations: int, maximum number of generations
    - elite_rate: float, fraction of population to preserve
    - mutation_rate: float, probability of mutating each gene

    Returns:
    - best_individual: list of floats, the best solution found
    """
    # Initialize population
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    # Determine number of elites
    elite_count = max(1, int(elite_rate * population_size))

    # Inspect fitness_function signature
    sig = inspect.signature(fitness_function)

    for gen in range(generations):
        # Evaluate fitness for each individual
        fitnesses = []
        for idx, individual in enumerate(population):
            if len(sig.parameters) == 2:
                fitness = fitness_function(individual, idx)
            else:
                fitness = fitness_function(individual)
            fitnesses.append(fitness)

        # Pair individuals with fitness and sort descending
        paired = list(zip(population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)

        # Update best overall solution
        top_ind, top_fit = paired[0]
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_individual = top_ind.copy()

        # Stop if reached target
        if best_fitness >= target_fitness:
            print(f"Target fitness {target_fitness} reached at generation {gen}.")
            break

        # Elitism: keep top performers
        elites = [ind for ind, fit in paired[:elite_count]]

        # Generate children to fill population
        children = []
        while len(children) < (population_size - elite_count):
            parent1, parent2 = random.sample(elites, 2)
            # Single-point crossover
            point = random.randint(1, individual_size - 1)
            child = parent1[:point] + parent2[point:]
            # Mutation: random reset per gene
            child = [
                gene if random.random() > mutation_rate else random.uniform(-1, 1)
                for gene in child
            ]
            children.append(child)

        # Form new population
        population = elites + children

    return best_individual
