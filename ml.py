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


def plots_for_feture0and1(X_data, weights_Data):
    # Wykres danych nieprzeważonych dla feature 0 i 1
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), width_ratios=[1, 1.2])
    fig.suptitle(
        "Gradient descent function effect on synthetics data (for feature 1 and 2)"
    )
    ax[0].scatter(
        X_data[y == 0][:, 0], X_data[y == 0][:, 1], color="blue", label="Class 0"
    )
    ax[0].scatter(
        X_data[y == 1][:, 0], X_data[y == 1][:, 1], color="red", label="Class 1"
    )
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].set_title("Unweighted data")
    ax[0].legend()

    # Wykres danych przeważonych dla feature 0 i 1
    ax[1].scatter(X_data[y == 0][:, 0], X_data[y == 0][:, 1])
    ax[1].scatter(X_data[y == 1][:, 0], X_data[y == 1][:, 1])
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].set_title("Weighted data")
    fig.colorbar(
        ax[1].scatter(X_data[:, 0], X_data[:, 1], c=weights_Data, cmap="viridis"),
        ax=ax[1],
        label="Weight",
    )

    plt.tight_layout()
    plt.savefig("Gradient descent effect.png")


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

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=seed
)

# Gradient descend execution
test_weights = gradient_descent(X_test, y_test)

# Grid-search
grid_test_weights = grid_search_scratch(X_test, y_test)
print(grid_test_weights)
# ---------------------------------------------------------------------------------------------
clf = SVC(gamma="auto")
clf.fit(X_train, y_train)

# Train another classifier with sample weighting
weighted_clf = SVC(gamma="auto")
weighted_clf.fit(X_train, y_train, sample_weight=gradient_descent(X_train, y_train))

# Train another one with grid search sample weighting
grid_weighted_clf = SVC(gamma="auto")
grid_weighted_clf.fit(
    X_train, y_train, sample_weight=grid_search_scratch(X_train, y_train)
)
# ---------------------------------------------------------------------------------------------

# Show prediction vector without weighted training
non_weighted_pred = clf.predict(X_test)

# Show prediction vector with weighted training
weighted_pred = weighted_clf.predict(X_test)
grid_weighted_pred = grid_weighted_clf.predict(X_test)

# Metrics for non-weighted samples model
f1_no_weights = f1_score(y_test, non_weighted_pred)
precision_no_weights = precision_score(y_test, non_weighted_pred)
recall_no_weights = recall_score(y_test, non_weighted_pred)
auc_roc_no_weights = roc_auc_score(y_test, non_weighted_pred)

# Metrics for weighted samples model
f1_with_weights = f1_score(y_test, weighted_pred)
precision_with_weights = precision_score(y_test, weighted_pred)
recall_with_weights = recall_score(y_test, weighted_pred)
auc_roc_with_weights = roc_auc_score(y_test, weighted_pred)

# Metrics for grid weighted samples model
f1_with_grid_weights = f1_score(y_test, grid_weighted_pred)
precision_with_grid_weights = precision_score(y_test, grid_weighted_pred)
recall_with_grid_weights = recall_score(y_test, grid_weighted_pred)
auc_roc_with_grid_weights = roc_auc_score(y_test, grid_weighted_pred)

rounding = 3

results = [
    [
        "Non-weighted",
        round(f1_no_weights, rounding),
        round(precision_no_weights, rounding),
        round(recall_no_weights, rounding),
        round(auc_roc_no_weights, rounding),
    ],
    [
        "Weighted",
        round(f1_with_weights, rounding),
        round(precision_with_weights, rounding),
        round(recall_with_weights, rounding),
        round(auc_roc_with_weights, rounding),
    ],
    [
        "Grid Weighted",
        round(f1_with_grid_weights, rounding),
        round(precision_with_grid_weights, rounding),
        round(recall_with_grid_weights, rounding),
        round(auc_roc_with_grid_weights, rounding),
    ],
]

headers = ["Model", "F1-score", "Precision", "Recall", "ROC-AUC"]

print(tabulate(results, headers=headers))

print()

# # Ploting input data vs. prediction
# # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# fig, ax = plt.subplots(2, 2, figsize=(23, 14), width_ratios=[1, 1.25,1,1])
# # ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
# ax[0].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0')
# ax[0].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1')
# ax[1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno', marker='o', label='Test samples')
# fig.colorbar(ax[1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap='inferno'), ax=ax[1], label="Weight")
#
# # ax[2].scatter(X_test[:, 0], X_test[:, 1], c=non_weighted_pred, cmap='coolwarm', marker='x', label='Unweighted prediction')
# ax[2].scatter(X_test[non_weighted_pred == 0][:, 0], X_test[non_weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
# ax[2].scatter(X_test[non_weighted_pred == 1][:, 0], X_test[non_weighted_pred == 1][:, 1], color='red', marker='o', label='Class 1')
#
# # ax[3].scatter(X_test[:, 0], X_test[:, 1], c=weighted_pred, cmap='coolwarm', marker='x', label='Weighted prediction')
# ax[3].scatter(X_test[weighted_pred == 0][:, 0], X_test[weighted_pred == 0][:, 1], color='blue', marker='x', label='Class 0')
# ax[3].scatter(X_test[weighted_pred == 1][:, 0], X_test[weighted_pred == 1][:, 1], color='red', marker='o', label='Class 1')
#
# ax[0].set_xlabel('Feature 1')
# ax[0].set_ylabel('Feature 2')
# ax[0].set_title('Unweighted data')
# ax[0].legend()
#
# ax[1].set_xlabel('Feature 1')
# ax[1].set_ylabel('Feature 2')
# ax[1].set_title('Data weights')
# ax[1].legend()
#
# ax[2].set_xlabel('Feature 1')
# ax[2].set_ylabel('Feature 2')
# ax[2].set_title('Unweighted prediction')
# ax[2].legend()
#
# ax[3].set_xlabel('Feature 1')
# ax[3].set_ylabel('Feature 2')
# ax[3].set_title('Weighted prediction')
# ax[3].legend()
# plt.tight_layout()
# plt.show()
#
# Ploting input data vs. prediction
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
fig, ax = plt.subplots(3, 2, figsize=(16, 14))

# ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='o', label='Test samples')
ax[0][0].scatter(
    X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color="blue", label="Class 0"
)
ax[0][0].scatter(
    X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color="red", label="Class 1"
)
ax[0][1].scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=test_weights,
    cmap="inferno",
    marker="o",
)
fig.colorbar(
    ax[0][1].scatter(X_test[:, 0], X_test[:, 1], c=test_weights, cmap="inferno"),
    label="Weight",
)

# ax[2].scatter(X_test[:, 0], X_test[:, 1], c=non_weighted_pred, cmap='coolwarm', marker='x', label='Unweighted prediction')
ax[1][0].scatter(
    X_test[non_weighted_pred == 0][:, 0],
    X_test[non_weighted_pred == 0][:, 1],
    color="blue",
    marker="x",
    label="Class 0",
)
ax[1][0].scatter(
    X_test[non_weighted_pred == 1][:, 0],
    X_test[non_weighted_pred == 1][:, 1],
    color="red",
    marker="x",
    label="Class 1",
)

# ax[3].scatter(X_test[:, 0], X_test[:, 1], c=weighted_pred, cmap='coolwarm', marker='x', label='Weighted prediction')
ax[1][1].scatter(
    X_test[weighted_pred == 0][:, 0],
    X_test[weighted_pred == 0][:, 1],
    color="blue",
    marker="x",
    label="Class 0",
)
ax[1][1].scatter(
    X_test[weighted_pred == 1][:, 0],
    X_test[weighted_pred == 1][:, 1],
    color="red",
    marker="x",
    label="Class 1",
)

ax[2][0].scatter(
    X_test[grid_weighted_pred == 0][:, 0],
    X_test[grid_weighted_pred == 0][:, 1],
    color="blue",
    marker="x",
    label="Class 0",
)
ax[2][0].scatter(
    X_test[grid_weighted_pred == 1][:, 0],
    X_test[grid_weighted_pred == 1][:, 1],
    color="red",
    marker="x",
    label="Class 1",
)
ax[2][1].scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=grid_test_weights,
    cmap="inferno",
    marker="o",
)
fig.colorbar(
    ax[2][1].scatter(X_test[:, 0], X_test[:, 1], c=grid_test_weights, cmap="inferno"),
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
plt.show()
