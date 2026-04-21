# Genetic Algorithm-Based Phishing Detection

Research reproduction project.
This repository contains a research-focused pipeline designed to enhance the detection of phishing websites. The project implements a hybrid feature selection strategy:

Statistical Prefilter: Uses Mutual Information to narrow down the feature space.

Genetic Algorithm (Wrapper): Evolves a population of feature subsets to find the most predictive combination for a downstream SVM classifier.

## Project Structure
The repository is organized into data, source code, and analysis notebooks:

``` Plaintext
PHISHING_GA_PAPER/
├── data/
│   ├── raw/                # Original UCI datasets (.arff, .csv)
│   └── processed/          # Cleaned, split, and scaled data for model input
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_ga_feature_selection.ipynb
│   └── 04_results_comparison.ipynb
├── src/                    # Core modular logic
│   ├── config.py           # Hyperparameters (Pop size, Mutation rate, etc.)
│   ├── evaluate.py         # Metrics calculation (Accuracy, F1, etc.)
│   ├── feature_prefilter.py# Mutual Information filtering logic
│   ├── ga_feature_selection.py # Core Genetic Algorithm implementation
│   ├── models.py           # Final model tuning via GridSearchCV
│   └── preprocess.py       # Data scaling and stratified splitting
├── main.py                 # Primary execution script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup & Installation
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### Create and Activate Virtual Environment:
```Bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Install Dependencies:

```Bash
pip install -r requirements.txt
```

### How to Run
1. Run the Full Pipeline
The main.py script executes the entire workflow, from loading processed data to outputting final model performance on the optimized feature set. 

``` Bash
python main.py 
``` 

#### Run with a different dataset:
To point the pipeline to a different dataset, look at the bottom of main.py within the if __name__ == "__main__": block.

You will find a variable named DATA_PATH. Simply update this string to the relative path of your new .pkl

``` python 
if __name__ == "__main__":
    # CHANGE THIS LINE to switch your dataset
    DATA_PATH = "data/processed/your_new_dataset.pkl"
    
```
### What happens in the pipeline:

Data Loading: Loads pre-split and scaled data from data/processed/uci_plf/.

Prefiltering: Reduces the feature set to the top K features (configured in config.py).

GA Optimization: Runs the Genetic Algorithm to find the best feature "chromosome." * Evaluation: Trains and tunes SVC, Logistic Regression, and Decision Tree models on the selected features and reports test set metrics.

2. Configuration
You can adjust the behavior of the GA (e.g., population size, mutation rate) by editing src/config.py. Key parameters include:

TOP_K_FEATURES: Number of features to keep after the initial statistical filter.

POP_SIZE & N_GENERATIONS: Control the "depth" and "breadth" of the GA search.

EARLY_STOP_PATIENCE: Prevents overfitting by stopping the GA if no improvement is seen.

### Eperimental Analysis
For a step-by-step breakdown of the research and visualization of the results (such as fitness curves), please refer to the files in the notebooks/ directory and our report. These are numbered sequentially to follow the research methodology.
