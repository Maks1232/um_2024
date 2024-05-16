import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from time import time
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
from ac_ml import print_metrics_for_npArray, calculate_metrics, calc_mean_metrics, plots_for_feture0and1

# # Set printoptions to max_size
# import sys
# np.set_printoptions(threshold=sys.maxsize)


# Data normalization from [X,Y] >> [0, 1]
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Gradient Descent wages calculation
def gradient_descent(_X, _y, learning_rate=0.1, w=4):
    num_samples, num_features = X.shape

    weights = np.ones(num_features) * w  # Future initialization
    weights_prew = weights

    finish = True
    while finish:
        cost = np.dot(weights, _X.T) - _y  # obliczanie kosztu
        gradient = np.dot(_X.T, cost) / num_samples  # obliczanie gradientu
        weights = weights_prew - (learning_rate * gradient)  # aktualizacja wag
        weights_prew = weights  # zachowanie aktualnych wag
        if (gradient[0] or gradient[1]) < 0.00001:
            finish = False  # koniec jeśli gradient jest bliski 0

    # sample_weights = normalize_data(np.dot(X, weights))  # obliczenie wag dla każdej próbki
    sample_weights = normalize_data(
        np.dot(_X, weights)
    )  # obliczenie wag dla każdej próbki

    return sample_weights


"""
Creating plot before and after calculating wages.
It is illustrative plot only for fist two features: 1 and 2.
"""


def import_dataset(file_path):
    try:
        df = pd.read_csv(file_path, delimiter="\t")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # From column 0 pick these rows which contains @input|@output and replace @input|@output to ''
    col_names = (
        df[df.iloc[:, 0].str.contains("@input|@output")]
        .iloc[:, 0]
        .str.replace("@input", "")
        .str.replace("@output", "")
    )
    # Concatenate two rows
    col_names = (
        np.array(col_names.str.split(","))[0] + np.array(col_names.str.split(","))[1]
    )
    # Remove rows with @
    df = df[df.iloc[:, 0].str.contains("@") == False]
    # Split one column to multiple ones
    df = df.iloc[:, 0].str.split(",", expand=True)
    # Assign column names
    df.columns = col_names
    # Reset index numbers
    df = df.reset_index(drop=True)
    # Due to error this option needs to be set
    pd.set_option("future.no_silent_downcasting", True)
    # Replace negative to 0 and positive to 1
    df = df.replace("negative", 0).replace("positive", 1)
    return df


def grid_search_scratch(xtrain, ytrain):

    # Weight vector [1, 1, 1, ...,1]
    weights = np.ones(len(xtrain))

    # For each sample >> maximize model metric
    for i in range(len(xtrain)):

        best_score = -np.inf
        t1 = time()
        # print("Sample {}/{} tuning time start: {}".format(i, len(X_train), t1))

        # Iteration with step 0.25
        for weight in np.arange(0.1, 1.1, 0.1):

            # Model initialization
            # model = SVC(probability=True)
            model = LogisticRegression()

            # Fit
            model.fit(xtrain, ytrain, sample_weight=weights)

            # Predication & Metric calculation
            # y_pred_proba = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(xtrain)
            score = roc_auc_score(ytrain, y_pred)

            # Score comparison
            if score > best_score:
                best_score = score
                weights[i] = weight

        if i % 10 == 0:
            t2 = time()
            print("Sample {}/{} tuning time end: {}".format(i, len(xtrain), t2))
            print("Diff: {}".format(t2 - t1))

    return weights


# df = import_dataset('dataset/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat')
# print(df)

seed = 135

# Generation imbalanced, synthetic dataset
X, y = make_classification(
    n_samples=100,
    weights=[0.9, 0.1],
    n_classes=2,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=seed,
)

# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=seed
# )

# # Gradient descend
# # Grid-search
# test_weights = gradient_descent(X_test, y_test)
# grid_test_weights = grid_search_scratch(X_test, y_test)

# ---------------------------------------------------------------------------------------------
n_splits = 2
n_repeats = 5
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=10)
results = np.zeros(shape=[4, n_splits * n_repeats, 4])

for i, (train_index, test_index) in enumerate(rkf.split(X, y)):

    clf = SVC(gamma="auto")
    clf.fit(X[train_index], y[train_index])
    y_pred = clf.predict(X[test_index])
    results[0, i, :] = (
        f1_score(y[test_index], y_pred),
        precision_score(y[test_index], y_pred, zero_division=1),
        recall_score(y[test_index], y_pred),
        roc_auc_score(y[test_index], y_pred),
    )

    weighted_clf = SVC(gamma="auto")
    train_weights = gradient_descent(X[train_index], y[train_index])
    weighted_clf.fit(X[train_index], y[train_index], sample_weight=train_weights)
    weighted_y_pred = weighted_clf.predict(X[test_index])
    results[1, i, :] = (
        f1_score(y[test_index], weighted_y_pred),
        precision_score(y[test_index], weighted_y_pred, zero_division=1),
        recall_score(y[test_index], weighted_y_pred),
        roc_auc_score(y[test_index], weighted_y_pred),
    )

    balanced_clf = SVC(gamma="auto", class_weight="balanced")
    balanced_clf.fit(X[train_index], y[train_index])
    balanced_y_pred = balanced_clf.predict(X[test_index])
    # print(balanced_y_pred)
    results[2, i, :] = (
        f1_score(y[test_index], balanced_y_pred),
        precision_score(y[test_index], balanced_y_pred, zero_division=1),
        recall_score(y[test_index], balanced_y_pred),
        roc_auc_score(y[test_index], balanced_y_pred),
    )

    grid_search_clf = SVC(gamma="auto")
    train_gs_weights = grid_search_scratch(X[train_index], y[train_index])
    grid_search_clf.fit(
        X[train_index], y[train_index], sample_weight=train_gs_weights
    )
    grid_search_y_pred = grid_search_clf.predict(X[test_index])
    results[3, i, :] = (
        f1_score(y[test_index], grid_search_y_pred),
        precision_score(y[test_index], grid_search_y_pred, zero_division=1),
        recall_score(y[test_index], grid_search_y_pred),
        roc_auc_score(y[test_index], grid_search_y_pred),
    )
