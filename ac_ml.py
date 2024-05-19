import random
from math import ceil
from nltk import accuracy
from scipy.stats import ttest_rel

from scipy import stats
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, \
    GridSearchCV
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from time import time

from tabulate import tabulate


def normalize_data(data):
    """
        Data normalization from [X,Y] >> [0, 1]
        Parameters:
            data: data to normalize
        Returns:
            normalized data in range [0,1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def gradient_descent(X, y, learning_rate=0.1, w=4):
    """
        Gradient Descent wages calculation function
        Parameters:
            X: feature matrix
            y: target list
            learning_rate: by default = 0.1
            w: by default = 4
        Returns:
            Calculated Gradient Descent wages
    """
    num_samples, num_features = X.shape

    weights = np.ones(num_features) * w  # Future initialization
    weights_prew = weights

    finish = True
    while finish:
        cost = np.dot(weights, X.T) - y  # cost calculation
        gradient = np.dot(X.T, cost) / num_samples  # gradient calculation
        weights = weights_prew - (learning_rate * gradient)  # weights update
        weights_prew = weights  # store actual weights
        if (gradient[0] or gradient[1]) < 0.00001:
            finish = False  # stop if gradient for class 0 or 1 is near zero

    sample_weights = np.dot(X, weights)  # calculating gradient for all observations

    return sample_weights





def plots_for_feture0and1(X_data, y_data, weights_Data, filename="Pictures\Gradient descent effect.png"):
    """
        Creating plot before and after calculating wages. It is illustrative plot only for fist two features: 1 and 2.
        Parameters:
            X_data: feature matrix
            y_data: target list
            weights_Data: weights for samples
            filename: name of the file in which plot will be saved
        Returns:
            void
    """
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
    """
        Function used to import dataset which is saved into .dat file
        Parameters:
            file_path: dataset .dat file location
        Returns:
            Imported dataset as DataFrame
    """
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
    # Concatenate these two rows
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
    # Replace negative to 0 and positive to 1
    df = df.replace("negative", 0).replace("positive", 1)
    # Change values from str to numeric
    df = df.apply(pd.to_numeric)
    return df


def calc_mean_metrics(array):
    """
        Function used to calculate mean metrics results
        Parameters:
            array: array with metrics results for each fold.
                    Array should have shape = [4 different classificators, n_splits * n_repeats, 4 metrics]
        Returns:
            DataFrame with mean results for each classificator and each metrics
    """
    rows_metric_name = ["F1-score", "Precision", "Recall", "ROC-AUC"]
    result = pd.DataFrame()
    for i in range(array.shape[0]):
        data = pd.DataFrame(array[i], columns=rows_metric_name)
        result = pd.concat([result, pd.DataFrame(data.mean())], axis=1)
    result.columns = ["Non-weighted", "Weighted", "Balanced", "Grid Search"]
    return result

def metrics_results(y_real, y_pred):
    """
        Function used to calculate metrics ["F1-score", "Precision", "Recall", "ROC-AUC"]
        Parameters:
            y_real: target list
            y_pred: predicted target list
        Returns:
            List of calculated metrics ["F1-score", "Precision", "Recall", "ROC-AUC"]
    """
    result = (f1_score(y_real, y_pred),
            precision_score(y_real, y_pred, zero_division=1),  # due to errors parameter zero_division=1 is mandatory
            recall_score(y_real, y_pred),
            roc_auc_score(y_real, y_pred),)
    return result

def calculate_metrics(X, y):
    """
        Function used to train classificators and calculate metrics for each one.
        First trained classificator is basic SVM. Second one uses sample_weight calculated by gradient_descent function.
        Third one is using hiperparameter class_weight="balanced".
        The last one uses sample_weight calculated by grid_search function.
        Parameters:
            X: feature matrix
            y: target list
        Returns:
            Calculated metrics ["F1-score", "Precision", "Recall", "ROC-AUC"] for each fold.
    """
    n_splits = 2
    n_repeats = 5
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=14)
    results = np.zeros(shape=[4, n_splits * n_repeats, 4])

    for i, (train_index, test_index) in enumerate(rkf.split(X, y)):
        clf = SVC(gamma="auto")
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        results[0, i, :] = metrics_results(y[test_index], y_pred)

        weighted_clf = SVC(gamma="auto")
        train_weights = gradient_descent(X[train_index], y[train_index])
        weighted_clf.fit(X[train_index], y[train_index], sample_weight=train_weights)
        weighted_y_pred = weighted_clf.predict(X[test_index])
        results[1, i, :] = metrics_results(y[test_index], weighted_y_pred)

        balanced_clf = SVC(gamma="auto", class_weight="balanced")
        balanced_clf.fit(X[train_index], y[train_index])
        balanced_y_pred = balanced_clf.predict(X[test_index])
        results[2, i, :] = metrics_results(y[test_index], balanced_y_pred)

        grid_search_clf = SVC(gamma="auto")
        train_gs_weights = grid_search(X[train_index], y[train_index])
        grid_search_clf.fit(X[train_index], y[train_index], sample_weight=train_gs_weights)
        grid_search_y_pred = grid_search_clf.predict(X[test_index])
        results[3, i, :] = metrics_results(y[test_index], grid_search_y_pred)

    return results


def grid_search(X, y, w=10):
    """
       Grid Search wages calculation function.
       Parameters:
           X: feature matrix
           y: target list
           w: by default = 10
       Returns:
           Founded wages
    """
    weights = np.ones(len(X)).astype(float) * w  # initialize weights
    """ tutaj można spróbować ewentualnie losowo inicjować wagi"""
    # weights = np.random.choice(np.arange(1, 51, 2), size=len(X))
    skf = StratifiedKFold(n_splits=3,shuffle=True, random_state=1)  # internal cross validation
    for train_index, test_index in skf.split(X, y):
        best_score = -np.inf  # initialize best metric score
        for i in train_index:  # loop through each element in train_index
            #"""to można potestować jak działa"""
        # for j in range(0, len(train_index), 15):
        #     i = train_index[j:j + 15]
            for weight in np.arange(1.0, 0.09, -0.1) * w:  # for each element test which wage gives better roc_auc_score
                model = tree.DecisionTreeClassifier()
                weights_temp = weights.copy()  # copy weights list to not overwrite it
                weights_temp[i] = weight  # substitute weight value and train model
                model.fit(X[train_index], y[train_index], sample_weight=weights_temp[train_index])
                y_pred = model.predict(X[test_index])
                score = roc_auc_score(y[test_index], y_pred)
                # if new checked weight value gives better roc_auc_score then
                if score > best_score:
                    best_score = score  # and save new best metric score
                    weights[i] = weight  # substitute weight
    print("Calculating...")
    return weights


# def grid_search_scratch(X, y):
#     weights = np.ones(len(X)).astype(float)
#     best_score = -np.inf
#     best_params = 0
#     balance = [{0: 0.1, 1: 1}, {0: 0.25, 1: 1}, {0: 0.5, 1: 1}, {0: 0.75, 1: 1}, {0: 0.1, 1: 1},
#                {0: 0.1, 1: 0.75}, {0: 0.25, 1: 0.75}, {0: 0.5, 1: 0.75}, {0: 0.75, 1: 0.75}, {0: 1, 1: 0.75},
#                {0: 0.1, 1: 0.5}, {0: 0.25, 1: 0.5}, {0: 0.5, 1: 0.5}, {0: 0.75, 1: 0.5}, {0: 1, 1: 0.5},
#                {0: 0.1, 1: 0.25}, {0: 0.25, 1: 0.25}, {0: 0.5, 1: 0.25}, {0: 0.75, 1: 0.25}, {0: 1, 1: 0.25},
#                {0: 0.1, 1: 0.1}, {0: 0.25, 1: 0.1}, {0: 0.5, 1: 0.1}, {0: 0.75, 1: 0.1},{0: 1, 1: 0.1},
#                ]
#     param_grid = dict(class_weight=balance)
#
#     skf = StratifiedKFold(n_splits=2)
#     for train_index, test_index in skf.split(X, y):
#         grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='roc_auc')
#         grid_result = grid.fit(X[train_index], y[test_index])
#         # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         if best_score < grid_result.best_score_:
#             best_score = grid_result.best_score_
#             best_params = grid_result.best_params_
#     for i in np.unique(y):
#         weights[np.where(y == i)[0]] = list(best_params.values())[0][i]
#     return weights

def t_student_results(metrics, filename):
    """
       Function used to perform T Student test and save results to .csv file.
       Parameters:
           metrics: obtained metrics from all folds
           filename: target csv filename
       Returns:
           void
    """
    # T student test for real data
    t_statystki = np.zeros(calc_mean_metrics(metrics).shape)
    p = t_statystki.copy()
    istotnosc_statystyczna = np.zeros(t_statystki.shape).astype(bool)
    przewaga = istotnosc_statystyczna.copy()
    alpha = 0.05

    roc_auc_scores_synth = metrics[:, :, -1]  # using roc auc score for t student test

    for i in range(roc_auc_scores_synth.shape[0]):
        for j in range(roc_auc_scores_synth.shape[0]):
            t_statystki[i, j], p[i, j] = stats.ttest_rel(roc_auc_scores_synth[i, :], roc_auc_scores_synth[j, :])
            if np.mean(roc_auc_scores_synth[i, :]) > np.mean(roc_auc_scores_synth[j, :]): przewaga[i, j] = True
            if p[i, j] < alpha: istotnosc_statystyczna[i, j] = True

    przewaga_istot_statyst = przewaga * istotnosc_statystyczna
    headers = ["Non-weighted", "Weighted", "Balanced", "Grid Search"]
    with open(filename, 'w') as plik:
        plik.write("Macierz t_statystki:\n")
        plik.write(tabulate(t_statystki, headers=headers, floatfmt=".8f"))
        plik.write("\n\nMacierz p:\n")
        plik.write(tabulate(p, headers=headers, floatfmt=".8f"))
        plik.write("\n\nMacierz przewagi (roc_auc_score):\n")
        plik.write(tabulate(przewaga, headers=headers))
        plik.write("\n\nIstotnosc statystyczna:\n")
        plik.write(tabulate(istotnosc_statystyczna, headers=headers))
        plik.write("\n\nPrzewaga istotna statystycznie:\n")
        plik.write(tabulate(przewaga_istot_statyst, headers=headers))





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

metrics_for_synth_data = calculate_metrics(X1, y1)
metrics_for_real_data = calculate_metrics(X2, y2)

with open("result.txt", "w") as f:
    print("Synthetic data", file=f)
    print(calc_mean_metrics(metrics_for_synth_data), file=f)
    print("\nReal data", file=f)
    print(calc_mean_metrics(metrics_for_real_data), file=f)

t_student_results(metrics_for_synth_data,'ttstudent_results_synthetic.csv')
t_student_results(metrics_for_real_data,'ttstudent_results_real.csv')

# X, y = make_classification(
#     n_samples=1000,
#     weights=[0.90, 0.10],
#     n_classes=2,
#     n_features=2,
#     n_redundant=0,
#     n_informative=1,
#     n_clusters_per_class=1,
#     random_state=3,
# )
# weights = gradient_descent(X, y)
# plots_for_feture0and1(X, y, weights, "Pictures/Gradient descent effect example.png")
# weights = grid_search_scratch(X, y)
# plots_for_feture0and1(X, y, weights, "Pictures/Grid search effect example.png")
