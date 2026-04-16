# ============================================================
# config.py — Central place to store all hyperparameters and
# experiment settings. 
# ============================================================

# generate same random numbers each run for reproducibility
SEED = 42

# set a fixed test size for all experiments to ensure fair comparisons
TEST_SIZE = 0.30

# ---- Prefilter ----
# get top-K most informative features so that the GA starts with a smaller, 
# more relevant search space
TOP_K_FEATURES = 50

# ---- Genetic Algorithm (GA) Parameters ----
# set population size to control how many candidate solutions 
# (feature subsets) are evaluated each generation
POP_SIZE = 50

# Max generations the GA is allowed to run before stopping
N_GENERATIONS = 1

# set mutation rate to control how often random changes are introduced to offspring
MUTATION_RATE = 0.01

# set crossover rate to control how often two parent chromosomes combine to create offspring
CROSSOVER_RATE = 0.80

# set tournament size for selection — how many candidates compete to be a parent
TOURNAMENT_SIZE = 3

# early stopping parameters to prevent overfitting and save time if the GA converges
EARLY_STOP_PATIENCE = 5

# The minimum fitness improvement required to reset the early stopping counter. If the best fitness doesn't improve by at least this amount for EARLY_STOP_PATIENCE generations, the GA will stop.
MIN_IMPROVEMENT = 0.001