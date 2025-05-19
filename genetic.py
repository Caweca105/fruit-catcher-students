import random
import inspect


def create_individual(individual_size, gene_min=0, gene_max=10):
    """
    Create a single individual represented by an integer vector of length `individual_size`.
    Genes are initialized uniformly at random as integers in [gene_min, gene_max].
    """
    return [random.randint(gene_min, gene_max) for _ in range(individual_size)]


def generate_population(individual_size, population_size, gene_min=0, gene_max=10):
    """
    Generate an initial population of integer individuals.

    Returns a list of individuals (lists of ints).
    """
    return [create_individual(individual_size, gene_min, gene_max) for _ in range(population_size)]


def genetic_algorithm(
    individual_size,
    population_size,
    fitness_function,
    target_fitness,
    generations,
    elite_rate=0.2,
    mutation_rate=0.05,
    gene_min=0,
    gene_max=10
):
    """
    Evolve a population of integer vectors via a genetic algorithm.

    Returns a tuple (best_individual, best_fitness).

    Parameters:
    - individual_size: int, number of genes per individual
    - population_size: int, number of individuals
    - fitness_function: callable(individual, seed?) -> float
    - target_fitness: float, threshold to stop early
    - generations: int, maximum number of generations
    - elite_rate: float in (0,1], fraction of population preserved
    - mutation_rate: float in (0,1], probability of mutating each gene
    - gene_min: int, minimum integer value for a gene
    - gene_max: int, maximum integer value for a gene
    """
    # Initialize population with integer genes
    population = generate_population(individual_size, population_size, gene_min, gene_max)
    best_individual = None
    best_fitness = float('-inf')
    elite_count = max(1, int(elite_rate * population_size))

    # Determine if fitness_function expects a seed
    sig = inspect.signature(fitness_function)
    use_seeds = (len(sig.parameters) == 2)
    seeds = [random.randrange(2**31) for _ in range(population_size)] if use_seeds else [None] * population_size

    # Evolution loop
    for gen in range(1, generations + 1):
        # Evaluate fitness for each individual
        fitnesses = []
        for idx, individual in enumerate(population):
            if use_seeds:
                fitness = fitness_function(individual, seeds[idx])
            else:
                fitness = fitness_function(individual)
            fitnesses.append(fitness)

        # Pair individuals with fitness (and seeds if used) and sort descending
        if use_seeds:
            paired = list(zip(population, fitnesses, seeds))
            paired.sort(key=lambda x: x[1], reverse=True)
        else:
            paired = list(zip(population, fitnesses))
            paired.sort(key=lambda x: x[1], reverse=True)

        # Update best overall solution
        top = paired[0]
        top_ind = top[0]
        top_fit = top[1]
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_individual = top_ind.copy()

        # Early stopping if target reached
        if best_fitness >= target_fitness:
            print(f"Target fitness {target_fitness} reached at generation {gen}.")
            break

        # Elitism: carry over top individuals (and seeds)
        if use_seeds:
            elites = paired[:elite_count]
            elites_inds = [ind for ind, fit, sd in elites]
            elites_seeds = [sd for ind, fit, sd in elites]
        else:
            elites_inds = [ind for ind, fit in paired[:elite_count]]

        # Generate children
        next_population = elites_inds.copy()
        next_seeds = elites_seeds.copy() if use_seeds else None
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(elites_inds, 2)
            # Single-point crossover
            point = random.randint(1, individual_size - 1)
            child = parent1[:point] + parent2[point:]
            # Mutation: reset to random int in [gene_min, gene_max]
            child = [
                gene if random.random() > mutation_rate else random.randint(gene_min, gene_max)
                for gene in child
            ]
            next_population.append(child)
            if use_seeds:
                next_seeds.append(random.randrange(2**31))

        # Prepare next generation
        population = next_population
        if use_seeds:
            seeds = next_seeds

    return best_individual, best_fitness
