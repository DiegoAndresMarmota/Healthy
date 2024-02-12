import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from archivos import make_dataset, plot, show_dataframe, plot_dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# The code `initial_dataset = make_dataset(100)` is calling a function called `make_dataset` and
# passing the argument `100` to it. This function is expected to generate a dataset, possibly with 100
# data points, and return it. The returned dataset is then assigned to the variable `initial_dataset`.
initial_dataset = make_dataset(100)
show_dataframe(initial_dataset)


# The code `figure = plot.figure()` creates a new figure object, which is like a blank canvas for
# plotting.
figure = plot.figure()
axis = figure.gca()
plot(axis, initial_dataset, "Initial Dataset")
plt.show()


# The code `scaler = StandardScaler()` creates an instance of the `StandardScaler` class from the
# `sklearn.preprocessing` module.
scaler = StandardScaler()
standard_scaler = scaler.fit_transform(initial_dataset)


def show_dataset( dataset, title, label, data) -> None:
    """
    The function "show_dataset" plots a dataset with a given title, label, and data, and displays the
    plot.
    
    :param dataset: The dataset parameter is the actual dataset that you want to visualize. It could be
    a pandas DataFrame, a numpy array, or any other data structure that contains the data you want to
    plot
    :param title: The title parameter is a string that represents the title of the dataset plot
    :param label: The label parameter is used to specify the label for the dataset in the plot. It is
    typically a string that describes the dataset or its purpose
    :param data: The `data` parameter is the actual data that you want to plot. It could be a list,
    array, or any other data structure that contains the values you want to visualize
    """
    try:
        plot_dataset(dataset, title, label, data)
        plt.show(dataset, title, label, data)
    except:
        pass
    finally:
        return show_dataset(
            initial_dataset, "Initial data", 
            standard_scaler, "Standardized data"
    )


def min_max_scaler():
    """
    The function `min_max_scaler` applies the MinMaxScaler transformation to a dataset and returns the
    scaled dataset.
    :return: the min-max scaled dataset.
    """
    try:
        scaler = MinMaxScaler()
        minmax_dataset = scaler.fit_transform(initial_dataset)
    except:
        pass
    finally:
        return show_dataset(
            initial_dataset, "Initial data",
            minmax_dataset, "MinMax data"
)

def scaler_max_abs():
    """
    The function `scaler_max_abs` applies the MaxAbsScaler transformation to a dataset and returns the
    scaled dataset.
    :return: the max-abs scaled dataset.
    """
    try:
        scaler = MaxAbsScaler()
        initial_dt = scaler.fit(initial_dataset)
        maxabs_dataset = scaler.transform(initial_dataset)
    except:
        pass
    finally:
        return show_dataset(
            initial_dt, "Initial data",
            maxabs_dataset, "MaxAbs data"
        ) 