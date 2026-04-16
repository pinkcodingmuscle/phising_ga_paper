import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# ================================================================
# ga_feature_selection.py
#
# This file implements a Genetic Algorithm (GA) for feature
# selection. The core idea is treating a feature subset as a
# "chromosome" of 0/1 bits, then evolving a population of them
# toward higher classification accuracy using operators inspired
# by biological evolution (selection, crossover, mutation).
# The GA is a wrapper method that evaluates feature subsets based
# on the performance of a downstream classifier (SVM in this case).
# ================================================================


def create_chromosome(n_features):
    # chromosome set to random 0/1 vector where 1 means "include this feature" 
    # and 0 means "exclude it"
    chromosome = np.random.randint(0, 2, size=n_features)

    # force at least one feature to be selected because an all-zero chromosome 
    # would be invalid for the classifier and cause errors during fitness 
    # evaluation. (simple repair step after random generation)
    if chromosome.sum() == 0:
        chromosome[np.random.randint(0, n_features)] = 1
    return chromosome


def create_population(pop_size, n_features):
    # generate a list of `pop_size` random chromosomes to form the 
    # initial population
    return [create_chromosome(n_features) for _ in range(pop_size)]


def fitness_function(chromosome, X, y, seed=42):
    # this function evaluates how good a given chromosome (feature subset) is by
    # measuring the cross-validated accuracy of an SVM trained on the selected 
    # features

    # decode the chromosome: get the indices of the features where the bit is 1
    selected_indices = np.where(chromosome == 1)[0]

    # if nofeatures selected, return worst possible fitness (0 accuracy) 
    # to discourage the GA from exploring empty subsets
    if len(selected_indices) == 0:
        return 0.0

    # Slice only the selected feature columns for this evaluation
    X_subset = X.iloc[:, selected_indices]

    # SVM with RBF kernel to capture non-linear relationships
    model = SVC(kernel="rbf", random_state=seed)

    # 3-fold cross-validation to get a reliable estimate of how well this 
    # feature subset generalizes.
    scores = cross_val_score(
        model,
        X_subset,
        y,
        cv=3,
        scoring="accuracy"
    )

    # Return mean accuracy across the 3 folds as the fitness scalar
    return scores.mean()


def tournament_selection(population, fitnesses, tournament_size=3):
    # This function implements tournament selection a method where we randomly select a few 
    # individuals from the population and pick the best among them to be a parent
    # for the next generation
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = selected_indices[0]  # start by assuming the first drawn is best

    # Walk through the rest of the tournament contestants.
    for idx in selected_indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx

    # Return a copy so mutations on the child cannot accidentally modify
    # the parent still sitting in the population list.
    return population[best_idx].copy()


def crossover(parent1, parent2, crossover_rate=0.8):
    # this function takes two parent chromosomes and produces two 
    # children by combining their genetic material at a randomly chosen point
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Choose a cut point to split the parents' gene sequences.
    point = np.random.randint(1, len(parent1) - 1)

    # Create children by swapping the tails of the parents at the cut point
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])

    # Repair by forcing one random gene on if the crossover somehow 
    # produced an all-zero child, which would be invalid for fitness evaluation
    if child1.sum() == 0:
        child1[np.random.randint(0, len(child1))] = 1
    if child2.sum() == 0:
        child2[np.random.randint(0, len(child2))] = 1

    # Return the two new child chromosomes
    return child1, child2


# this function applies random mutations to a chromosome by flipping bits 
# with a certain probability
def mutate(chromosome, mutation_rate=0.01):
    # create copy of chromosome to mutate so we don't modify the original in-place
    mutated = chromosome.copy()

    # Iterate through each gene and flip it with probability equal to mutation_rate
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # flip the bit -> 0 to 1 (add feature) or 1 to 0 (remove feature).
            mutated[i] = 1 - mutated[i]

    # randomly flip bits if mutation resulted in an all-zero chromosome
    # this would be invalid for fitness evaluation
    if mutated.sum() == 0:
        mutated[np.random.randint(0, len(mutated))] = 1

    return mutated


def run_ga(
    X,
    y,
    pop_size=50,
    n_generations=100,
    mutation_rate=0.01,
    crossover_rate=0.80,
    tournament_size=3,
    early_stop_patience=5,
    min_improvement=0.001,
    seed=42
):
    # random seed for reproducibility of the GA's stochastic 
    # processes (initial population, selection, crossover, mutation)
    np.random.seed(seed)

    # start with a randomly generated population of candidate 
    # solutions (feature subsets)
    population = create_population(pop_size, X.shape[1])

    # track the best and average fitness of each generation for analysis and plotting
    best_fitness_history = []
    avg_fitness_history = []

    best_chromosome = None # store best solution found across all generations
    best_fitness = -np.inf # initialize to worst possible fitness
    no_improvement_count = 0  # counter for early stopping

    # loop through generations of evolution
    for generation in range(n_generations):
        # evaluate the fitness of each chromosome in the current population
        fitnesses = [fitness_function(ch, X, y, seed=seed) for ch in population]

        # Identify the best-performing chromosome this generation
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_chromosome = population[gen_best_idx].copy()
        gen_avg_fitness = np.mean(fitnesses)

        # Record history for plotting
        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        # update the global best solution if this generation's best 
        # is better than the best we've seen so far
        if gen_best_fitness - best_fitness > min_improvement:
            best_fitness = gen_best_fitness
            best_chromosome = gen_best_chromosome.copy()
            no_improvement_count = 0  # reset the patience counter
        else:
            no_improvement_count += 1  # no real progress in this generation

        print(
            f"Generation {generation + 1:3d} | "
            f"Best: {gen_best_fitness:.4f} | Avg: {gen_avg_fitness:.4f}"
        )

        # prevent overfitting and save time if the GA converges by stopping early 
        # if no improvement has been seen for a number of consecutive generations
        if no_improvement_count >= early_stop_patience:
            print("Early stopping triggered.")
            break

        #build new generation through selection, crossover, and mutation
        new_population = []

        while len(new_population) < pop_size:
            # select two parents from the current population using 
            # tournament selection
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            # crossover - combine parent gene sequences.
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # mutation - small random perturbations to each child
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)
            # add child2 only if we still have room in the new population 
            # to avoid exceeding pop_size
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Replace the old population with the new generation of offspring
        population = new_population

    # convert the best chromosome back to feature names for 
    # interpretability
    selected_indices = np.where(best_chromosome == 1)[0]
    selected_features = X.columns[selected_indices].tolist()

    return best_chromosome, best_fitness, selected_features, best_fitness_history, avg_fitness_history