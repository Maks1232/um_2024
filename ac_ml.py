import random

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from time import time


# Data normalization from [X,Y] >> [0, 1]
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Gradient Descent wages calculation
def gradient_descent(X, y, learning_rate=0.1, w=4):
    num_samples, num_features = X.shape

    weights = np.ones(num_features) * w  # Future initialization
    weights_prew = weights

    finish = True
    while finish:
        cost = np.dot(weights, X.T) - y  # obliczanie kosztu
        gradient = np.dot(X.T, cost) / num_samples  # obliczanie gradientu
        weights = weights_prew - (learning_rate * gradient)  # aktualizacja wag
        weights_prew = weights  # zachowanie aktualnych wag
        if (gradient[0] or gradient[1]) < 0.00001:
            finish = False  # koniec jeśli gradient jest bliski 0

    sample_weights = normalize_data(
        np.dot(X, weights)
    )  # obliczenie wag dla każdej próbki

    return sample_weights


def grid_search_scratch(X_train, y_train):

    # Weight vector [1, 1, 1, ...,1]
    weights = np.ones(len(X_train))

    # For each sample >> maximize model metric
    for i in range(len(X_train)):

        best_score = -np.inf
        t1 = time()
        # print("Sample {}/{} tuning time start: {}".format(i, len(X_train), t1))

        # Iteration with step 0.25
        for weight in np.arange(0.1, 1.1, 0.1):

            # Model initialization
            # model = SVC(probability=True)
            model = LogisticRegression()

            # Fit
            model.fit(X_train, y_train, sample_weight=weights)

            # Predication & Metric calculation
            # y_pred_proba = model.predict_proba(X_train)[:, 1]
            y_pred = model.predict(X_train)
            score = roc_auc_score(y_train, y_pred)

            # Score comparison
            if score > best_score:
                best_score = score
                weights[i] = weight

        if i % 10 == 0:
            t2 = time()
            print("Sample {}/{} tuning time end: {}".format(i, len(X_train), t2))
            print("Diff: {}".format(t2 - t1))

    return weights


"""
Creating plot before and after calculating wages.
It is illustrative plot only for fist two features: 1 and 2.
"""


def plots_for_feture0and1(
    X_data, y_data, weights_Data, filename="Gradient descent effect.png"
):
    # Wykres danych nieprzeważonych dla feature 0 i 1
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1, 1.2])
    fig.suptitle(
        "Gradient descent function effect on synthetics data (for feature 1 and 2)"
    )
    ax[0].scatter(
        X_data[y_data == 0][:, 0],
        X_data[y_data == 0][:, 1],
        color="blue",
        label="Class 0",
    )
    ax[0].scatter(
        X_data[y_data == 1][:, 0],
        X_data[y_data == 1][:, 1],
        color="red",
        label="Class 1",
    )
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].set_title("Unweighted data")
    ax[0].legend()

    # Wykres danych przeważonych dla feature 0 i 1
    ax[1].scatter(X_data[y_data == 0][:, 0], X_data[y_data == 0][:, 1])
    ax[1].scatter(X_data[y_data == 1][:, 0], X_data[y_data == 1][:, 1])
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].set_title("Weighted data")
    fig.colorbar(
        ax[1].scatter(X_data[:, 0], X_data[:, 1], c=weights_Data, cmap="viridis"),
        ax=ax[1],
        label="Weight",
    )

    plt.tight_layout()
    plt.savefig(filename)


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
    # Change values from str to numeric
    df = df.apply(pd.to_numeric)
    return df


def calc_mean_metrics(array):
    rows_name = ["F1-score", "Precision", "Recall", "ROC-AUC"]
    result = pd.DataFrame()
    for i in range(array.shape[0]):
        data = pd.DataFrame(array[i], columns=rows_name)
        result = pd.concat([result, pd.DataFrame(data.mean())], axis=1)
    result.columns = ["Non-weighted", "Weighted", "Balanced", "Grid Search"]
    return result


def print_metrics_for_npArray(array):
    if len(array) == 4:
        coulmns_name = ["F1-score", "Precision", "Recall", "ROC-AUC"]
        for i in range(array.shape[0]):
            data = pd.DataFrame(array[i], columns=coulmns_name)
            print(data)
    else:
        print("Array is not 4 dimensional")


def calculate_metrics(X, y):
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

    return results


X1, y1 = make_classification(
    n_samples=1000,
    weights=[0.90, 0.10],
    n_classes=2,
    n_features=8,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    random_state=10,
)
df = import_dataset("dataset/yeast-0-2-5-7-9_vs_3-6-8/yeast-0-2-5-7-9_vs_3-6-8.dat")
X2 = np.array(df.iloc[:, :-1])
y2 = np.array(df.iloc[:, -1])

with open("r1.txt", "w") as f:
    print("Synthetic data", file=f)
    metrics_for_synth_data = calculate_metrics(X1, y1)
    # print_metrics_for_3dim_npArray(metrics_for_synth_data)
    print(calc_mean_metrics(metrics_for_synth_data), file=f)

    print("\nReal data", file=f)
    metrics_for_real_data = calculate_metrics(X2, y2)
    # print_metrics_for_3dim_npArray(metrics_for_real_data)
    print(calc_mean_metrics(metrics_for_real_data), file=f)


# grid search effect example
# seed = random.randint(1,100)
# print(seed)
# Gradient descent effect example
X, y = make_classification(
    n_samples=1000,
    weights=[0.90, 0.10],
    n_classes=2,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    random_state=3,
)
weights = gradient_descent(X, y)
plots_for_feture0and1(X, y, weights, "../ml_2024/um_2024/Pictures/Gradient descent effect example.png")
weights = grid_search_scratch(X, y)
plots_for_feture0and1(X, y, weights, "../ml_2024/um_2024/Pictures/Grid search effect example.png")


# def plot_weighted_and_nonweighted():
#
#     fig, ax = plt.subplots(2, 2, figsize=(16, 14))
#
#     ax[0][0].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0')
#     ax[0][0].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1')
#     ax[0][1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno', marker='o', label='Test samples')
#     fig.colorbar(ax[0][1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno'), label="Weight")
#
#     ax[1][0].scatter(X_test[non_weighted_pred == 0][:, 0], X_test[non_weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
#     ax[1][0].scatter(X_test[non_weighted_pred == 1][:, 0], X_test[non_weighted_pred == 1][:, 1], color='red', marker='x', label='Class 1')
#
#     ax[1][1].scatter(X_test[weighted_pred == 0][:, 0], X_test[weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
#     ax[1][1].scatter(X_test[weighted_pred == 1][:, 0], X_test[weighted_pred == 1][:, 1], color='red', marker='x', label='Class 1')
#
#     ax[0][0].set_xlabel('Feature 1')
#     ax[0][0].set_ylabel('Feature 2')
#     ax[0][0].set_title('Unweighted data')
#     ax[0][0].legend()
#
#     ax[0][1].set_xlabel('Feature 1')
#     ax[0][1].set_ylabel('Feature 2')
#     ax[0][1].set_title('Data weights')
#     ax[0][1].legend()
#
#     ax[1][0].set_xlabel('Feature 1')
#     ax[1][0].set_ylabel('Feature 2')
#     ax[1][0].set_title('Unweighted prediction')
#     ax[1][0].legend()
#
#     ax[1][1].set_xlabel('Feature 1')
#     ax[1][1].set_ylabel('Feature 2')
#     ax[1][1].set_title('Weighted prediction')
#     ax[1][1].legend()
#     plt.tight_layout()
#     plt.savefig("Unweighted and weighted prediction.png")
