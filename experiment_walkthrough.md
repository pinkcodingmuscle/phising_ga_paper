# Experiment Walkthrough — Phishing Detection with Genetic Algorithm Feature Selection

This document walks through the entire experiment from start to finish, explaining
what every piece of active code does and why decisions were made the way they were.
It is written from the perspective of a student working through the reasoning for
the first time.

---

## High-Level Goal

The experiment asks: *can a Genetic Algorithm (GA) find a small, high-quality
subset of features from a phishing-detection dataset that lets an SVM classify
URLs as phishing or legitimate at least as well as using all features?*

The pipeline has three main stages:

1. **Data preparation** — load the raw dataset, inspect it, split into train/test,
   and scale features.
2. **Feature pre-filtering** — use Mutual Information to quickly discard the least
   useful features before handing off to the GA.
3. **GA feature selection** — evolve a population of candidate feature subsets
   toward higher SVM cross-validation accuracy, then save the winners.

---

## File Map

```
src/config.py               — all hyperparameters in one place
src/preprocess.py           — train/test split + MinMax scaling
src/feature_prefilter.py    — Mutual Information pre-filter
src/ga_feature_selection.py — the full Genetic Algorithm
notebooks/01_dataset_inspection.ipynb  — Stage 1: data prep
notebooks/03_ga_feature_selection.ipynb — Stages 2 & 3: MI filter + GA
```

---

## Stage 0 — Configuration (`src/config.py`)

Before any code runs, all tunable numbers live in one file so that changing a
value in one place updates every notebook automatically.

| Constant | Value | What it controls |
|---|---|---|
| `SEED` | `42` | Makes every random operation reproducible across runs |
| `TEST_SIZE` | `0.30` | 30 % of data is held out for final evaluation |
| `TOP_K_FEATURES` | `50` | How many features survive the MI pre-filter |
| `POP_SIZE` | `50` | Number of candidate feature subsets alive at once |
| `N_GENERATIONS` | `100` | Maximum number of evolution cycles |
| `MUTATION_RATE` | `0.01` | Per-gene probability of a random bit flip |
| `CROSSOVER_RATE` | `0.80` | Probability that two parents actually swap genes |
| `TOURNAMENT_SIZE` | `3` | How many individuals compete in each selection round |
| `EARLY_STOP_PATIENCE` | `5` | Stop if no improvement for this many consecutive generations |
| `MIN_IMPROVEMENT` | `0.001` | Minimum fitness gain that counts as real progress |

---

## Stage 1 — Data Preparation (`notebooks/01_dataset_inspection.ipynb`)

### 1.1 Imports

```python
import pandas as pd
import pickle
import sys
import os
from scipy.io import arff
sys.path.append("..")
from src.config import SEED, TEST_SIZE
from src.preprocess import split_data, scale_data
```

The notebook lives inside `notebooks/`, one level below the project root where
`src/` lives. `sys.path.append("..")` walks one directory up so that Python can
find and import from the `src` package. Without it, every `from src.xxx import`
line would raise a `ModuleNotFoundError`.

---

### 1.2 Loading the Data

```python
# Dataset 1 — ARFF format (Weka/OpenML)
data, meta = arff.loadarff('../data/raw/Phishing_Legitimate_full.arff')
df = pd.DataFrame(data)
df.head()

# Dataset 2 — plain CSV
df = pd.read_csv('../data/raw/dataset_full.csv')
print(df.shape)
df.head()
```

Two datasets are loaded. Dataset 1 uses the ARFF format (Weka's standard). The
`arff.loadarff()` function returns a structured NumPy array (`data`) and a
metadata object (`meta`). The array is converted to a pandas DataFrame so that
column names, `.head()`, `.info()`, and other familiar methods are available.

> **ARFF quirk:** string and nominal values come out as Python *bytes* (e.g.
> `b'1'`), not plain strings. This affects the target column and must be fixed
> before the label is used.

Dataset 2 (`dataset_full.csv`) is a plain CSV so `pd.read_csv()` handles it
directly. `df` is overwritten with Dataset 2 because the rest of the pipeline is
dataset-agnostic — swapping the load call is all it takes to switch datasets.

---

### 1.3 Data Inspection

```python
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
```

Four quick checks before any modelling:

- **`shape`** — confirms how many rows (samples) and columns (features) the
  dataset has.
- **`info()`** — shows the data type of every column. Any column showing
  `dtype=object` instead of a numeric type would need conversion.
- **`isnull().sum()`** — counts missing values per column. Missing values would
  require an imputation strategy (e.g. filling with the column mean) before
  training.
- **`duplicated().sum()`** — counts identical rows. Duplicates can inflate
  training data and skew model learning by repeating the same examples.

---

### 1.4 Separating Features and Target (`src/preprocess.py` reference)

```python
target_col = "CLASS_LABEL"
X = df.drop(columns=[target_col])
y = df["CLASS_LABEL"].astype(str).astype(int)

print("X shape:", X.shape[1])
print("y shape:", y.shape)
print(y.value_counts())
print(y.unique())
```

In supervised learning the data is always split into:

- **X** — the feature matrix (everything the model learns from).
- **y** — the target vector (the labels the model is trying to predict:
  `1` = phishing, `0` = legitimate).

The label column is dropped from `df` to produce `X`. The ARFF byte-string issue
is handled by chaining two `astype()` calls: `bytes → str → int`. Going directly
from bytes to int raises a `ValueError`.

`y.value_counts()` reveals class balance. Severely imbalanced classes (e.g. 90 %
phishing) would mean even a naive model that always predicts the majority class
scores very high accuracy — important to know before interpreting results.

---

### 1.5 Train / Test Split (`src/preprocess.py — split_data`)

```python
# src/preprocess.py
def split_data(X, y, test_size=0.30, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )
    return X_train, X_test, y_train, y_test
```

```python
# notebook call
X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, seed=SEED)
print(X_train.shape, X_test.shape)
```

30 % of the data is held out as a test set the model *never* sees during training.
This simulates real-world deployment: the model will encounter new, unseen URLs
and we measure performance on those.

`stratify=y` is critical — it preserves the original class ratio in both splits.
Without it, random chance might put nearly all phishing examples into training,
making the test set unrepresentative and evaluation scores misleading.

`random_state=seed` makes the exact same split happen every time the notebook is
re-run. Without a fixed seed, two runs would produce different splits and results
would not be directly comparable.

---

### 1.6 Feature Scaling (`src/preprocess.py — scale_data`)

```python
# src/preprocess.py
def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
```

```python
# notebook call
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
print(X_train_scaled.describe().loc[["min", "max"]])
```

Many ML algorithms — especially SVM — are distance-based. A feature measured in
thousands would dominate one measured in single digits purely because of scale,
regardless of its actual predictive value. `MinMaxScaler` maps every feature into
the `[0, 1]` range, putting them all on equal footing.

**The most important rule here:** `fit_transform()` is called *only on the
training data*. The scaler learns the minimum and maximum values from `X_train`
and then `transform()` applies those same parameters to `X_test`. If we fit the
scaler on the test set too, the test set's statistics would leak into the pipeline
(*data leakage*), artificially inflating evaluation scores.

The result is wrapped back in a pandas `DataFrame` (rather than left as a NumPy
array) so that column names survive — downstream code that selects features by
name depends on this.

---

### 1.7 Saving to Disk

```python
with open("../data/processed/dataset1_split_scaled.pkl", "wb") as f:
    pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test), f)
```

`pickle` serialises the four arrays into a single binary file. Downstream
notebooks load this file instead of re-running the entire preparation pipeline.
This matters because the GA notebook is slow — decoupling preprocessing from
experimentation means changing a GA hyperparameter does not force a re-split of
the data.

---

## Stage 2 — Mutual Information Pre-filter (`notebooks/03_ga_feature_selection.ipynb` + `src/feature_prefilter.py`)

### 2.1 Loading the Saved Data

```python
with open("../data/processed/dataset1_split_scaled.pkl", "rb") as f:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(f)
```

Reload exactly the same split the preprocessing notebook produced. Using the same
partition across all experiments ensures every model is evaluated on the same
held-out examples.

---

### 2.2 Why Pre-filter?

With `N` features there are `2^N` possible subsets. Even with only 50 features,
`2^50 ≈ 10^15` — completely infeasible to evaluate exhaustively. The GA explores
this space *heuristically* (intelligently, without checking everything), but a
smaller search space still means faster convergence.

Mutual Information (MI) provides a fast, filter-based first pass. It scores
features individually by measuring how much knowing a feature's value reduces
uncertainty about the class label. Features with near-zero MI are essentially
independent of the label — they add noise rather than signal.

---

### 2.3 Pre-filter Code (`src/feature_prefilter.py — prefilter_features`)

```python
def prefilter_features(X_train, X_test, y_train, k=50, seed=42):
    k = min(k, X_train.shape[1])

    mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
    mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    selected_features = mi_series.head(k).index.tolist()

    X_train_filtered = X_train[selected_features].copy()
    X_test_filtered = X_test[selected_features].copy()

    return X_train_filtered, X_test_filtered, selected_features, mi_series
```

```python
# notebook call
X_train_ga, X_test_ga, selected_prefilter_features, mi_series = prefilter_features(
    X_train_scaled, X_test_scaled, y_train,
    k=TOP_K_FEATURES, seed=SEED
)
print("Features before prefilter:", X_train_scaled.shape[1])
print("Features after prefilter:", X_train_ga.shape[1])
mi_series.head(20)
```

Step by step:

1. `k = min(k, X_train.shape[1])` — guard against requesting more features than
   exist in the dataset.
2. `mutual_info_classif()` returns one score per feature. It is computed only on
   training data to prevent test leakage.
3. The scores are placed into a named pandas `Series` and sorted descending so the
   most informative features come first.
4. The top `k` feature names are extracted and used to slice both `X_train` and
   `X_test`. Using `.copy()` creates independent DataFrames rather than views,
   which avoids a common pandas `SettingWithCopyWarning` later.
5. The full `mi_series` is returned too so the notebook can display rankings as a
   sanity check.

**Known limitation:** MI scores features independently. It does not detect
redundancy — two highly correlated features can both score high even though
together they add little more information than one alone. The GA's combinatorial
search handles this in the next stage.

---

## Stage 3 — Genetic Algorithm (`src/ga_feature_selection.py`)

The GA is the core of this project. It is inspired by biological evolution: a
population of candidates (feature subsets) competes, the fittest reproduce, and
their offspring gradually improve over generations.

Each candidate is a **chromosome** — a binary vector with one bit per feature.
`1` means "include this feature", `0` means "exclude it". The GA's job is to find
the chromosome whose selected features give the SVM the highest cross-validated
accuracy.

---

### 3.1 Creating a Chromosome

```python
def create_chromosome(n_features):
    chromosome = np.random.randint(0, 2, size=n_features)
    if chromosome.sum() == 0:
        chromosome[np.random.randint(0, n_features)] = 1
    return chromosome
```

A chromosome is initialised by drawing a random `0` or `1` for each feature
position. The all-zero check is a *repair step* — an all-zero chromosome means no
features are selected, which would crash the SVM downstream. If that happens,
one random gene is forced to `1`.

---

### 3.2 Creating the Initial Population

```python
def create_population(pop_size, n_features):
    return [create_chromosome(n_features) for _ in range(pop_size)]
```

`POP_SIZE = 50` chromosomes are generated randomly. Starting with a diverse
random population is intentional — the GA needs to explore broadly before it
starts exploiting the best solutions it has found.

---

### 3.3 Fitness Function

```python
def fitness_function(chromosome, X, y, seed=42):
    selected_indices = np.where(chromosome == 1)[0]
    if len(selected_indices) == 0:
        return 0.0
    X_subset = X.iloc[:, selected_indices]
    model = SVC(kernel="rbf", random_state=seed)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring="accuracy")
    return scores.mean()
```

The fitness function answers *"how good is this feature subset?"* It:

1. Decodes the chromosome to find which feature columns are selected.
2. Slices those columns from the training data.
3. Trains an SVM with an RBF kernel on the subset using 3-fold cross-validation.
4. Returns the mean accuracy across the 3 folds as a single number.

3-fold CV is used instead of a single train/validation split to get a more
reliable accuracy estimate. It also guards against the GA overfitting its feature
selection to one lucky random split. `cv=3` rather than `cv=5` is a speed
compromise — this function is called `POP_SIZE × N_GENERATIONS` times in total.

---

### 3.4 Tournament Selection

```python
def tournament_selection(population, fitnesses, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = selected_indices[0]
    for idx in selected_indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx].copy()
```

Selection decides which chromosomes get to reproduce. Tournament selection randomly
picks `TOURNAMENT_SIZE = 3` chromosomes from the population and returns the one
with the highest fitness. Better chromosomes are more likely to be chosen, but
weaker ones still have a chance — this balances *exploitation* (using what works)
with *exploration* (trying new things).

A `.copy()` is returned so that mutations applied to the child do not accidentally
modify the parent that is still sitting in the population list.

---

### 3.5 Crossover

```python
def crossover(parent1, parent2, crossover_rate=0.8):
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    if child1.sum() == 0:
        child1[np.random.randint(0, len(child1))] = 1
    if child2.sum() == 0:
        child2[np.random.randint(0, len(child2))] = 1
    return child1, child2
```

Single-point crossover picks a random cut point in the chromosome and swaps the
tails of the two parents to produce two children. This models biological
recombination — children inherit traits from both parents, hopefully combining the
best of each (e.g. parent 1's good early features + parent 2's good late features).

With probability `1 - CROSSOVER_RATE = 0.20`, crossover is skipped and the
parents pass through unchanged. This preserves some solutions without modification.

The all-zero repair is applied to children too — crossover could theoretically
produce a zero-only child if both parents had `0` in complementary positions.

---

### 3.6 Mutation

```python
def mutate(chromosome, mutation_rate=0.01):
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    if mutated.sum() == 0:
        mutated[np.random.randint(0, len(mutated))] = 1
    return mutated
```

Mutation randomly flips individual bits with probability `MUTATION_RATE = 0.01`
(1 % per gene). A `0` becomes a `1` (add a feature) or a `1` becomes a `0`
(remove one). This introduces small random changes that can escape local optima
that crossover alone cannot escape — essentially the GA's exploration mechanism.

The same all-zero repair is applied after mutation.

---

### 3.7 The Main Evolution Loop (`run_ga`)

```python
np.random.seed(seed)
population = create_population(pop_size, X.shape[1])

best_fitness_history = []
avg_fitness_history = []
best_chromosome = None
best_fitness = -np.inf
no_improvement_count = 0

for generation in range(n_generations):
    fitnesses = [fitness_function(ch, X, y, seed=seed) for ch in population]

    gen_best_idx = np.argmax(fitnesses)
    gen_best_fitness = fitnesses[gen_best_idx]
    gen_best_chromosome = population[gen_best_idx].copy()
    gen_avg_fitness = np.mean(fitnesses)

    best_fitness_history.append(gen_best_fitness)
    avg_fitness_history.append(gen_avg_fitness)

    if gen_best_fitness - best_fitness > min_improvement:
        best_fitness = gen_best_fitness
        best_chromosome = gen_best_chromosome.copy()
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stop_patience:
        print("Early stopping triggered.")
        break

    new_population = []
    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, fitnesses, tournament_size)
        parent2 = tournament_selection(population, fitnesses, tournament_size)
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.append(child1)
        if len(new_population) < pop_size:
            new_population.append(child2)

    population = new_population

selected_indices = np.where(best_chromosome == 1)[0]
selected_features = X.columns[selected_indices].tolist()
return best_chromosome, best_fitness, selected_features, best_fitness_history, avg_fitness_history
```

Each generation follows the same cycle:

| Step | What happens |
|---|---|
| **Evaluate** | Every chromosome in the population gets a fitness score via SVM cross-validation |
| **Track best** | The global best chromosome is updated if this generation improved by more than `MIN_IMPROVEMENT` |
| **Early stop check** | If no improvement for `EARLY_STOP_PATIENCE` generations, halt |
| **Select** | Two parents are chosen via tournament selection |
| **Crossover** | Parents combine to produce two children |
| **Mutate** | Small random bit flips are applied to each child |
| **Replace** | The old population is entirely replaced by the new offspring |

`best_fitness` is initialised to `-np.inf` so that any real fitness value (even
0.0) is guaranteed to be an improvement on the first generation.

The `no_improvement_count` counter is reset to `0` whenever a genuine improvement
occurs and incremented otherwise. When it reaches `EARLY_STOP_PATIENCE = 5`, the
loop breaks early — this avoids wasting compute time on generations that are
unlikely to improve the result.

After the loop, the best chromosome's `1` positions are decoded back into feature
names using the DataFrame's column index.

---

### 3.8 Running the GA (notebook call)

```python
best_chromosome, best_fitness, selected_features, best_hist, avg_hist = run_ga(
    X_train_ga, y_train,
    pop_size=POP_SIZE,
    n_generations=N_GENERATIONS,
    mutation_rate=MUTATION_RATE,
    crossover_rate=CROSSOVER_RATE,
    tournament_size=TOURNAMENT_SIZE,
    early_stop_patience=EARLY_STOP_PATIENCE,
    min_improvement=MIN_IMPROVEMENT,
    seed=SEED
)
print("Best fitness:", best_fitness)
print("Selected feature count:", len(selected_features))
print(selected_features)
```

All hyperparameters come from `config.py`. The GA returns the best chromosome,
its fitness score, the names of the selected features, and the per-generation
history lists.

---

### 3.9 Convergence Plot

```python
plt.figure(figsize=(10, 5))
plt.plot(best_hist, label="Best Fitness")
plt.plot(avg_hist, label="Average Fitness")
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("GA Fitness Over Generations")
plt.legend()
plt.show()
```

The convergence plot is one of the most important diagnostics for a GA:

- **Best Fitness curve** — should rise quickly in early generations and then
  plateau as the population converges on a good solution.
- **Average Fitness curve** — represents the population as a whole. A large gap
  between the two curves means the population is still diverse (exploring). When
  the average tracks the best closely, the population has converged.

Things that indicate problems: best fitness never rising (stuck from the start),
average fitness dropping over time (mutation rate too high, destroying good
solutions), or the two curves being identical from generation 1 (population
collapsed with no diversity).

---

### 3.10 Saving GA Results

```python
with open("../data/processed/dataset1_selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

with open("../data/processed/dataset1_ga_history.pkl", "wb") as f:
    pickle.dump((best_hist, avg_hist), f)
```

Two files are saved:

- **`dataset1_selected_features.pkl`** — the list of feature names chosen by the
  GA. Notebook 04 loads this to slice `X_train` / `X_test` when evaluating the
  final model on the held-out test set.
- **`dataset1_ga_history.pkl`** — the per-generation best and average fitness
  lists. Saved so the convergence plot can be reproduced in notebook 04 without
  re-running the expensive GA.

---

## Summary of the Full Data Flow

```
raw .arff / .csv
      │
      ▼
01_dataset_inspection.ipynb
  └─ load → inspect → split (stratified 70/30) → MinMax scale → save .pkl
                                                                      │
                                                                      ▼
03_ga_feature_selection.ipynb
  └─ load .pkl
       └─ Mutual Information prefilter (keep top 50)
            └─ Genetic Algorithm (50 chromosomes × up to 100 generations)
                 └─ Fitness = SVM 3-fold CV accuracy on selected features
                      └─ Tournament selection → Crossover → Mutation
                           └─ save selected_features.pkl + ga_history.pkl
                                                                      │
                                                                      ▼
                                              04_results_comparison.ipynb
                                                (model evaluation — TBD)
```
