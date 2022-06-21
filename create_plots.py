import torch
import sys
import numpy as np
import matplotlib
import pickle
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path


# set some variables
PATH_OUTPUT = Path(__file__).parent.absolute()
PATH_NN = PATH_OUTPUT.parent / "save_nn"
PATH_DATA = PATH_OUTPUT.parent / "Data"
PATH_PLOTS = PATH_OUTPUT / "example_plots"

# some plot settings
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.figsize'] = (16., 8.)
dpi = 150

plt.rcParams.update({
    'lines.linewidth': 1.0,
    'lines.linestyle': '-',
    'lines.color': 'black',
    'font.family': 'serif',
    'font.weight': 'bold',  # normal
    'font.size': 16.0,  # 10.0
    'text.color': 'black',
    'text.usetex': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.labelweight': 'bold',  # normal
    'axes.labelcolor': 'black',
    'axes.formatter.limits': [-4, 4],
    'xtick.major.size': 7,
    'xtick.minor.size': 4,
    'xtick.major.pad': 8,
    'xtick.minor.pad': 8,
    'xtick.labelsize': 'x-large',
    'xtick.minor.width': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.size': 7,
    'ytick.minor.size': 4,
    'ytick.major.pad': 8,
    'ytick.minor.pad': 8,
    'ytick.labelsize': 'x-large',
    'ytick.minor.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.numpoints': 1,
    'legend.fontsize': 'x-large',
    'legend.shadow': False,
    'legend.frameon': False})


def standard_deviation(arr):
    """
    Calculate the standard deviation

    Args:
        arr: array of values

    Returns:
        the calculated standard deviation as array
    """
    return np.sqrt(np.sum((arr - np.mean(arr))**2) / len(arr))


def standard_scaler(arr):
    """
    Calculate the standard scale

    Args:
        arr: array of values

    Returns:
        calculated standard scale as array
    """
    return (arr - np.mean(arr)) / standard_deviation(arr)


def inversion(arr, standard_scaled):
    """

    Args:
        arr: array of values
        standard_scaled: calculated standard scale as array

    Returns:
        calculated inverse of the standard scale

    """
    return standard_scaled * standard_deviation(arr) + np.mean(arr)


def load_simulated_data(filename, batch_size=64):
    """
    Load the Data from *.pkl-file

    Args:
        filename: path of the *.pkl-file as string
        use_simulated_data: True if simulated data for training
        balrog_year: Choose balrog year for training, default is Y3
        length_of_data: How many training data want to use? For all set value to -1
        batch_size: size of batch

    Returns: dictionary with list of training data, list of test data, used deep field data and used flux data
    """

    # start = -40
    # stop = -3

    # open file
    infile = open(filename, 'rb')    # filename

    # load pickle as pandas dataframe
    df = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    # if use_simulated_data:
    pd_deep_field_data = df["deep field"]
    pd_flux_data = df["flux"]

    # calculate the standard scale of deep field to avoid huge numbers
    pd_standardized_deep_field = standard_scaler(pd_deep_field_data)
    pd_standardized_flux = standard_scaler(pd_flux_data)

    # create train and test arrays
    pd_train_deep_field = pd_standardized_deep_field[:int(len(pd_standardized_deep_field) * .8)]
    pd_train_flux = pd_standardized_flux[:int(len(pd_standardized_flux) * .8)]
    pd_test_deep_field = pd_standardized_deep_field[int(len(pd_standardized_deep_field) * .8):]
    pd_test_flux = pd_standardized_flux[int(len(pd_standardized_flux) * .8):]

    lst_gandalf_training = []
    for idx, deep_field_value in enumerate(pd_train_deep_field):
        lst_gandalf_training.append({
            "deep field value scaled": deep_field_value,
            "deep field value": inversion(pd_deep_field_data, deep_field_value),
            "flux value scaled": pd_train_flux[idx],
            "flux value": inversion(pd_flux_data, pd_train_flux[idx])
        })

    lst_gandalf_test = []
    for idx, deep_field_value in enumerate(pd_test_deep_field):
        lst_gandalf_test.append({
            "deep field value scaled": deep_field_value,
            "deep field value": inversion(pd_deep_field_data, deep_field_value),
            "flux value scaled": pd_test_flux[idx+int(len(pd_standardized_flux) * .8)],
            "flux value": inversion(pd_flux_data, pd_test_flux[idx+int(len(pd_standardized_flux) * .8)])
        })

    return {
        "training data": lst_gandalf_training,
        "test data": lst_gandalf_test,
        "deep field data": pd_deep_field_data,
        "flux data": pd_flux_data
    }


def main(path_neural_net_folder, path_test_data, path_save_figure=None, figure_name=None, show_figure=False):
    """

    Args:
        path_neural_net_folder:
        path_test_data:
        path_save_figure:
        figure_name:
        show_figure:

    Returns:

    """
    lst_balrogs = []

    if os.path.exists(path_neural_net_folder):
        for neural_net in os.listdir(path_neural_net_folder):
            if "balrog" in neural_net:
                balrog_name = str(neural_net).split("_")[1]
                lr = balrog_name.replace(".pt", "")
                lst_balrogs.append([
                    torch.load(path_neural_net_folder/neural_net, map_location=torch.device('cpu')),
                    lr])

    # Load the data
    dict_loaded_data = load_simulated_data(filename=path_test_data)

    for balrog_item in lst_balrogs:
        # Init new lists
        lst_deep_field = []
        lst_flux = []
        lst_result = []
        lst_standard_deviation = []

        balrog = balrog_item[0]
        lr = balrog_item[1]
        for idx, test_item in enumerate(dict_loaded_data["test data"]):
            result = balrog.forward(torch.randn(1), torch.FloatTensor([test_item["deep field value scaled"]]))
            lst_result.append(inversion(dict_loaded_data["flux data"], result[0].item()))
            lst_deep_field.append(test_item["deep field value"])
            lst_flux.append(test_item["flux value"])
            lst_standard_deviation.append(np.sqrt((result[0].item() - test_item["flux value scaled"]) ** 2))

        # Calculate accuracy and the mean of the results
        accuracy = 100 * (1 - np.mean(lst_standard_deviation))

        # Print results of the test
        print("accuracy", accuracy)

        # Creating a dictionary and pandas data frame with the results
        dict_result = {'result': lst_result, 'flux': lst_flux, "deep field": lst_deep_field}
        df = pd.DataFrame(dict_result)

        # Plotting the results if True
        # Plot the deep field against flux and against result

        plt.plot(df["deep field"], df["flux"], ".b", label="balrog data from february")
        plt.plot(df["deep field"], df["result"], ".r", label=f"generator result with accuracy of {accuracy:3.2f}%")
        plt.xlabel("deep field flux")
        plt.ylabel("true flux")
        plt.title(f"{figure_name.replace('_', ' ')}; learning rate {lr}")
        plt.legend()
        if path_save_figure is not None:
            # create folder
            if not os.path.exists(path_save_figure):
                os.mkdir(path_save_figure)
            plt.savefig(f"{path_save_figure}/{figure_name}_lr_{lr}.png", dpi=100)
        if show_figure is True:
            plt.show()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    path_nn_folder = PATH_NN / "Simulated_data_function_2_with_noise_20000_training_examples"
    path_data = PATH_DATA / "simulated_data_function_2_with_noise_20000.pkl"
    fig_name = "simulated_data_function_with_noise"

    main(
        path_neural_net_folder=path_nn_folder,
        path_test_data=path_data,
        path_save_figure=PATH_PLOTS,
        figure_name=fig_name,
        show_figure=False
    )
