import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.dirname(__file__))
from helper_functions import standard_scaler, standard_deviation


def load_simulated_data(filename):
    """
    Load the Data from *.pkl-file

    Args:
        filename: path of the *.pkl-file as string

    Returns: dictionary with list of training data, list of test data, used deep field data and used flux data
    """

    # open file
    infile = open(filename, 'rb')    # filename

    # load pickle as pandas dataframe
    df = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    # if use_simulated_data:
    deep_field_mean = df["deep field"].mean()
    deep_field_std_dev = standard_deviation(df["deep field"].to_numpy(), deep_field_mean)
    deep_field_scaled = standard_scaler(df["deep field"].to_numpy(), deep_field_mean, deep_field_std_dev)

    wide_field_mean = df["flux"].mean()
    wide_field_std_dev = standard_deviation(df["flux"].to_numpy(), wide_field_mean)
    wide_field_scaled = standard_scaler(df["flux"].to_numpy(), wide_field_mean, wide_field_std_dev)

    arr_train_deep_field_scaled = deep_field_scaled[:int(len(df["deep field"]) * .8)]
    arr_train_wide_field_scaled = wide_field_scaled[:int(len(df["flux"]) * .8)]

    arr_test_deep_field_scaled = deep_field_scaled[int(len(df["deep field"]) * .8):]
    arr_test_wide_field_scaled = wide_field_scaled[int(len(df["flux"]) * .8):]

    dict_train_data = {
        "deep field": df["deep field"].to_numpy()[:int(len(df["deep field"]) * .8)],
        "wide field": df["flux"].to_numpy()[:int(len(df["flux"]) * .8)],
        "deep field scaled": arr_train_deep_field_scaled,
        "wide field scaled": arr_train_wide_field_scaled,
        "deep field mean": deep_field_mean,
        "deep field standard deviation": deep_field_std_dev,
        "wide field mean": wide_field_mean,
        "wide field standard deviation": wide_field_std_dev
    }
    dict_test_data = {
        "deep field": df["deep field"].to_numpy()[int(len(df["deep field"]) * .8):],
        "wide field": df["flux"].to_numpy()[int(len(df["deep field"]) * .8):],
        "deep field scaled": arr_test_deep_field_scaled,
        "wide field scaled": arr_test_wide_field_scaled,
        "deep field mean": deep_field_mean,
        "deep field standard deviation": deep_field_std_dev,
        "wide field mean": wide_field_mean,
        "wide field standard deviation": wide_field_std_dev
    }
    return dict_train_data, dict_test_data


def main(filename):
    """"""
    train_data, test_data = load_simulated_data(filename=filename)
    print(train_data["deep field"])


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"\..\Data"
    filename = filepath + r"\norm_simulated_data_function_different_noise_5000.pkl"
    main(filename)
