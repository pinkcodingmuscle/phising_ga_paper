import pandas as pd
import sys
import os
from scipy.io import arff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src import config
from src.feature_prefilter import prefilter_features
from src.ga_feature_selection import run_ga
from src.models import evaluate_final_models
from src.preprocess import split_data, scale_data # Re-added for ARFF files
from src.evaluate import evaluate_model 

# Ensure the 'src' directory is in the path if running from the root
sys.path.append(os.path.join(os.getcwd(), 'src'))

def main(X_train, X_test, y_train, y_test):
    """
    Main function to run the feature selection and model evaluation pipeline.
    
    Args:
        X_train (pd.DataFrame): Preprocessed training features.
        X_test (pd.DataFrame): Preprocessed testing features.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
    """
    print("--- Starting Feature Selection Pipeline ---")
    
    # 1. Statistical Prefilter
    print(f"Prefiltering features to top {config.TOP_K_FEATURES}...")
    X_train_pre, X_test_pre, top_features_list, mi_scores = prefilter_features(
        X_train, X_test, y_train, 
        k=config.TOP_K_FEATURES, 
        seed=config.SEED
    )
    print(f"Features remaining after prefilter: {len(top_features_list)}")

    # 2. Genetic Algorithm (Wrapper Phase)
    print("Running Genetic Algorithm (this may take a few minutes)...")
    best_chrom, best_fitness, selected_features, best_hist, avg_hist = run_ga(
        X_train_pre, y_train,
        pop_size=config.POP_SIZE,
        n_generations=config.N_GENERATIONS,
        mutation_rate=config.MUTATION_RATE,
        crossover_rate=config.CROSSOVER_RATE,
        tournament_size=config.TOURNAMENT_SIZE,
        early_stop_patience=config.EARLY_STOP_PATIENCE,
        min_improvement=config.MIN_IMPROVEMENT,
        seed=config.SEED
    )
    
    print(f"GA complete. Selected {len(selected_features)} optimal features.")
    print(f"Best Validation Accuracy during GA: {best_fitness:.4f}")

    # 3. Final Model Comparison and Tuning
    # Train multiple models on the optimized subset found by the GA
    X_train_final = X_train_pre[selected_features]
    X_test_final = X_test_pre[selected_features]
    
    print("\nTuning final models (SVC, Logistic Regression, Decision Tree)...")
    results_df = evaluate_final_models(X_train_final, y_train)
    
    #print("\n--- Final Model Comparison (Cross-Validation Results) ---")
    #print(results_df[['model', 'best_score', 'best_params']])

    #return selected_features, results_df
    print("\n--- Final Test Set Evaluation (Unseen Data) ---")
    # Example: Take the best model from your tuning (e.g., the first one in results_df)
    # This loop runs your evaluate_model function for the best found estimator
    final_test_metrics = []
    
    # Assuming evaluate_final_models returns a dataframe with 'model_object'
    for index, row in results_df.iterrows():
        metrics = evaluate_model(
            row['model'], 
            row['best_estimator'], # Ensure evaluate_final_models returns the actual estimator
            X_train_final, 
            X_test_final, 
            y_train, 
            y_test
        )
        final_test_metrics.append(metrics)
    
    final_performance = pd.DataFrame(final_test_metrics)
    print(final_performance)

    return selected_features, final_performance

if __name__ == "__main__":
    # Adjust this path to match your local setup
    DATA_PATH = "data/processed/uci_small/uci_small_split_scaled.pkl"
    
    if os.path.exists(DATA_PATH):
        # Unpack the pre-split and pre-scaled data
        X_train, X_test, y_train, y_test = pd.read_pickle(DATA_PATH)
        
        # Execute the pipeline
        final_features, comparison_results = main(X_train, X_test, y_train, y_test)
    else:
        print(f"Error: Processed data not found at {DATA_PATH}")