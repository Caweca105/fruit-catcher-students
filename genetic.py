import random
import inspect


def create_individual(individual_size, gene_min=-1, gene_max=1):
    """
    Create a single individual represented by an integer vector of length `individual_size`.
    Genes are initialized uniformly at random as integers in [gene_min, gene_max].
    """
    return [random.randint(gene_min, gene_max) for _ in range(individual_size)]


def generate_population(individual_size, population_size, gene_min=-1, gene_max=1):
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
    elite_rate=2,
    mutation_rate=5,
    gene_min=-1,
    gene_max=1
):
    """
    Evolve a population of integer vectors via a genetic algorithm.

    Parameters:
    - individual_size: int, number of genes per individual
    - population_size: int, number of individuals in population
    - fitness_function: callable, computes fitness from individual (and optional seed)
    - target_fitness: float, fitness threshold for stopping
    - generations: int, maximum number of generations
    - elite_rate: int, number of elites to preserve each generation (must be >=1)
    - mutation_rate: int, percentage (0-100) chance to mutate each gene
    - gene_min: int, minimum integer value for a gene
    - gene_max: int, maximum integer value for a gene

    Returns:
    - best_individual: list of ints, the best solution found
    """
    # Initialize population with integer genes
    population = generate_population(individual_size, population_size, gene_min, gene_max)
    best_individual = None
    best_fitness = float('-inf')

    # Determine number of elites to carry over
    elite_count = max(1, int(elite_rate))

    # Check if fitness_function accepts a seed parameter
    sig = inspect.signature(fitness_function)
    use_seeds = (len(sig.parameters) == 2)
    seeds = [random.randrange(2**31) for _ in range(population_size)] if use_seeds else [None] * population_size

    for gen in range(generations):
        # Evaluate fitness for each integer individual
        fitnesses = []
        for idx, individual in enumerate(population):
            if use_seeds:
                fitness = fitness_function(individual, seeds[idx])
            else:
                fitness = fitness_function(individual)
            fitnesses.append(fitness)

        # Pair individuals with fitness (and seeds if used), sort descending
        if use_seeds:
            paired = list(zip(population, fitnesses, seeds))
            paired.sort(key=lambda x: x[1], reverse=True)
        else:
            paired = list(zip(population, fitnesses))
            paired.sort(key=lambda x: x[1], reverse=True)

        # Update best overall solution
        top = paired[0]
        top_fit = top[1]
        top_ind = top[0]
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_individual = top_ind.copy()

        # Terminate if reached target fitness
        if best_fitness >= target_fitness:
            print(f"Target fitness {target_fitness} reached at generation {gen}.")
            break

        # Elitism: retain top performers
        if use_seeds:
            elites = paired[:elite_count]
            elites_ind = [ind for ind, fit, sd in elites]
            elites_seeds = [sd for ind, fit, sd in elites]
        else:
            elites_ind = [ind for ind, fit in paired[:elite_count]]

        # Generate integer children to refill population
        children = []
        children_seeds = []
        while len(children) < (population_size - elite_count):
            # Select two parents at random
            p1, p2 = random.sample(elites_ind, 2)
            # Single-point crossover
            point = random.randint(1, individual_size - 1)
            child = p1[:point] + p2[point:]
            # Mutation: random reset per gene (integer)
            mutated = [
                gene if random.randint(1, 100) > mutation_rate else random.randint(gene_min, gene_max)
                for gene in child
            ]
            children.append(mutated)
            if use_seeds:
                children_seeds.append(random.randrange(2**31))

        # Form new population
        population = elites_ind + children
        if use_seeds:
            seeds = elites_seeds + children_seeds

    return best_individual
