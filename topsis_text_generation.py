import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize the decision matrix
def normalize_matrix(X):
    norm_X = X / np.sqrt((X**2).sum(axis=0))
    return norm_X

# Calculate the ideal and negative ideal solutions
def calculate_ideal_solutions(norm_X, criteria):
    ideal_best = np.max(norm_X, axis=0) if criteria == 1 else np.min(norm_X, axis=0)
    ideal_worst = np.min(norm_X, axis=0) if criteria == 1 else np.max(norm_X, axis=0)
    return ideal_best, ideal_worst

# Calculate the Euclidean distance to the ideal solutions
def calculate_distances(norm_X, ideal_best, ideal_worst):
    distance_to_best = np.sqrt(((norm_X - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((norm_X - ideal_worst) ** 2).sum(axis=1))
    return distance_to_best, distance_to_worst

# TOPSIS Score
def calculate_topsis_score(distance_to_best, distance_to_worst):
    score = distance_to_worst / (distance_to_best + distance_to_worst)
    return score

# Function to apply TOPSIS
def apply_topsis(input_file, output_file):
    # Read CSV
    df = pd.read_csv(input_file)

    # Extract the decision matrix (Perplexity, BLEU, ROUGE)
    X = df.iloc[:, 1:].values

    # Normalize the decision matrix
    norm_X = normalize_matrix(X)

    # Define criteria for each column (0: cost, 1: benefit)
    criteria = [0, 1, 1]  # Perplexity is cost, BLEU and ROUGE are benefits

    # Calculate ideal and negative ideal solutions
    ideal_best, ideal_worst = calculate_ideal_solutions(norm_X, criteria)

    # Calculate distances
    distance_to_best, distance_to_worst = calculate_distances(norm_X, ideal_best, ideal_worst)

    # Calculate TOPSIS score
    topsis_scores = calculate_topsis_score(distance_to_best, distance_to_worst)

    # Convert the scores to a pandas Series and rank them
    df['TOPSIS Score'] = topsis_scores
    df['Rank'] = pd.Series(topsis_scores).rank(ascending=False)

    # Save the results to a new CSV file
    df.to_csv(output_file, index=False)

    # Print results for verification
    print(df)

    # Visualization: Barplot of TOPSIS Scores
    sns.barplot(x='Models', y='TOPSIS Score', data=df)
    plt.title('TOPSIS Ranking of Text Generation Models')
    plt.show()

    print(f'Results saved to {output_file}')

# Input and Output files
input_file = 'input_text_generation_models.csv'
output_file = 'topsis_text_generation_results.csv'

# Apply TOPSIS
apply_topsis(input_file, output_file)
