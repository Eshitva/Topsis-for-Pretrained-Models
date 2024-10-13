import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_matrix(X):
    norm_X = X / np.sqrt((X**2).sum(axis=0))
    return norm_X

def calculate_ideal_solutions(norm_X, criteria):
    ideal_best = np.max(norm_X, axis=0) if criteria == 1 else np.min(norm_X, axis=0)
    ideal_worst = np.min(norm_X, axis=0) if criteria == 1 else np.max(norm_X, axis=0)
    return ideal_best, ideal_worst

def calculate_distances(norm_X, ideal_best, ideal_worst):
    distance_to_best = np.sqrt(((norm_X - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((norm_X - ideal_worst) ** 2).sum(axis=1))
    return distance_to_best, distance_to_worst

def calculate_topsis_score(distance_to_best, distance_to_worst):
    score = distance_to_worst / (distance_to_best + distance_to_worst)
    return score

def apply_topsis(input_file, output_file):
    df = pd.read_csv(input_file)

    X = df.iloc[:, 1:].values

    norm_X = normalize_matrix(X)

    criteria = [0, 1, 1]  
    ideal_best, ideal_worst = calculate_ideal_solutions(norm_X, criteria)

    distance_to_best, distance_to_worst = calculate_distances(norm_X, ideal_best, ideal_worst)

    topsis_scores = calculate_topsis_score(distance_to_best, distance_to_worst)

    df['TOPSIS Score'] = topsis_scores
    df['Rank'] = pd.Series(topsis_scores).rank(ascending=False)

    df.to_csv(output_file, index=False)

    print(df)

    sns.barplot(x='Models', y='TOPSIS Score', data=df)
    plt.title('TOPSIS Ranking of Text Generation Models')
    plt.show()

    print(f'Results saved to {output_file}')

input_file = 'input_text_generation_models.csv'
output_file = 'topsis_text_generation_results.csv'

apply_topsis(input_file, output_file)
