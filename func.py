import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
