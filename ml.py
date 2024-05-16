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
from func import print_metrics_for_npArray, calc_mean_metrics, plots_for_feture0and1

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


def grid_search_scratch(xtrain, ytrain, xtest, ytest):

    # Weight vector [1, 1, 1, ...,1]
    weights = np.ones(len(xtrain))

    # For each sample >> maximize model metric
    for i in range(len(xtrain)):

        best_score = -np.inf
        t1 = time()
        # print("Sample {}/{} tuning time start: {}".format(i, len(X_train), t1))

        # Iteration with step 0.25
        for weight in np.arange(0.1, 1.1, 0.25):

            # Model initialization
            model = SVC(probability=True)
            # model = LogisticRegression()

            # Fit
            model.fit(xtrain, ytrain, sample_weight=weights)

            # Predication & Metric calculation
            # y_pred_proba = model.predict_proba(X_train)[:, 1]
            pred = model.predict(xtest)
            score = f1_score(ytest, pred)

            # Score comparison
            if score > best_score:
                best_score = score
                weights[i] = weight

        if i % 50 == 0:
            t2 = time()
            print("Sample {}/{} tuning time end: {}".format(i, len(xtrain), t2))
            print("Diff: {}".format(t2 - t1))

    return weights


# df = import_dataset('dataset/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat')
# print(df)

seed = 135

# Generation imbalanced, synthetic dataset
X, y = make_classification(
    n_samples=500,
    weights=[0.9, 0.1],
    n_classes=2,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=seed,
)

# ---------------------------------------------------------------------------------------------
n_splits = 2
n_repeats = 5
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=10)
results = np.zeros(shape=[4, n_splits * n_repeats, 4])
# folder_path = os.path.join(os.path.expanduser("~"), "Pictures")

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
    test_weights = gradient_descent(X[test_index], y[test_index])

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

    train_gs_weights = grid_search_scratch(X[train_index], y[train_index], X[test_index], y[test_index])
    test_gs_weights = grid_search_scratch(X[train_index], y[train_index], X[test_index], y[test_index])
    print(test_gs_weights)
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
    if i == 1:
        fig, ax = plt.subplots(3, 2, figsize=(16, 14))

        # ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
        ax[0][0].scatter(
            X[test_index][y[test_index] == 0][:, 0], X[test_index][y[test_index] == 0][:, 1], color="blue", label="Class 0"
        )
        ax[0][0].scatter(
            X[test_index][y[test_index] == 1][:, 0], X[test_index][y[test_index] == 1][:, 1], color="red", label="Class 1"
        )
        ax[0][1].scatter(
            X[test_index][:, 0],
            X[test_index][:, 1],
            c=test_weights,
            cmap="inferno",
            marker="o",
        )
        fig.colorbar(
            ax[0][1].scatter(X[test_index][:, 0], X[test_index][:, 1], c=test_weights, cmap="inferno"),
            label="Weight",
        )

        # ax[2].scatter(X_test[:, 0], X_test[:, 1], c=non_weighted_pred, cmap='coolwarm', marker='x', label='Unweighted prediction')
        ax[1][0].scatter(
            X[test_index][y_pred == 0][:, 0],
            X[test_index][y_pred == 0][:, 1],
            color="blue",
            marker="x",
            label="Class 0",
        )
        ax[1][0].scatter(
            X[test_index][y_pred == 1][:, 0],
            X[test_index][y_pred == 1][:, 1],
            color="red",
            marker="x",
            label="Class 1",
        )

        # ax[3].scatter(X_test[:, 0], X_test[:, 1], c=weighted_pred, cmap='coolwarm', marker='x', label='Weighted prediction')
        ax[1][1].scatter(
            X[test_index][weighted_y_pred == 0][:, 0],
            X[test_index][weighted_y_pred == 0][:, 1],
            color="blue",
            marker="x",
            label="Class 0",
        )
        ax[1][1].scatter(
            X[test_index][weighted_y_pred == 1][:, 0],
            X[test_index][weighted_y_pred == 1][:, 1],
            color="red",
            marker="x",
            label="Class 1",
        )

        ax[2][0].scatter(
            X[test_index][grid_search_y_pred == 0][:, 0],
            X[test_index][grid_search_y_pred == 0][:, 1],
            color="blue",
            marker="x",
            label="Class 0",
        )
        ax[2][0].scatter(
            X[test_index][grid_search_y_pred == 1][:, 0],
            X[test_index][grid_search_y_pred == 1][:, 1],
            color="red",
            marker="x",
            label="Class 1",
        )
        ax[2][1].scatter(
            X[test_index][:, 0],
            X[test_index][:, 1],
            c=test_gs_weights,
            cmap="inferno",
            marker="o",
        )
        fig.colorbar(
            ax[2][1].scatter(X[test_index][:, 0], X[test_index][:, 1], c=test_gs_weights, cmap="inferno"),
            label="Weight",
        )

        ax[0][0].set_xlabel("Feature 1")
        ax[0][0].set_ylabel("Feature 2")
        ax[0][0].set_title("Unweighted data")
        ax[0][0].legend()

        ax[0][1].set_xlabel("Feature 1")
        ax[0][1].set_ylabel("Feature 2")
        ax[0][1].set_title("Data weights")
        ax[0][1].legend()

        ax[1][0].set_xlabel("Feature 1")
        ax[1][0].set_ylabel("Feature 2")
        ax[1][0].set_title("Unweighted prediction")
        ax[1][0].legend()

        ax[1][1].set_xlabel("Feature 1")
        ax[1][1].set_ylabel("Feature 2")
        ax[1][1].set_title("Weighted prediction")
        ax[1][1].legend()

        ax[2][0].set_xlabel("Feature 1")
        ax[2][0].set_ylabel("Feature 2")
        ax[2][0].set_title("Grid Weighted prediction")
        ax[2][0].legend()

        ax[2][1].set_xlabel("Feature 1")
        ax[2][1].set_ylabel("Feature 2")
        ax[2][1].set_title("Grid search weights")
        ax[2][1].legend()

        plt.tight_layout()
        plt.savefig("6_plots.png")
        plt.show()
avg_results = np.mean(results, axis=1)
df = pd.DataFrame(avg_results, columns=["F1", "Precision", "Recall", "Roc-Auc"])
metric_names = ["Non-weighted", "Weighted", "Balanced", "Grid Search"]
df.insert(0, "Mode", metric_names)

# print(df)
print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


# weights = gradient_descent(X, y)
# plots_for_feture0and1(X, y, weights, "Pictures/Gradient descent effect example.png")
# weights = grid_search_scratch(X, y)
# plots_for_feture0and1(X, y, weights, "Pictures/Grid search effect example.png")