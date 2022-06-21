import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle


def load_simulated_data(filename, balrog_year="Y3", length_of_data=-1, batch_size=64):
    """
    Load the Data from *.pkl-file

    Args:
        filename: path of the *.pkl-file as string
        balrog_year: Choose balrog year for training, default is Y3
        length_of_data: How many training data want to use? For all set value to -1
        batch_size: size of batch

    Returns: dictionary with list of training data, list of test data, used deep field data and used flux data
    """

    # open file
    infile = open(filename, 'rb')
    # load pickle as pandas dataframe
    df = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    for key in df:
        print(key)

    print("################################ deep_balrog ##############################################################")

    infile2 = open("deep_balrog.pkl", 'rb')
    # load pickle as pandas dataframe
    df2 = pd.DataFrame(pickle.load(infile2, encoding='latin1'))

    # close file
    infile2.close()

    for key in df2:
        print(key)
    exit()

    for key in df:
        print(key)
    exit()

    # Set dec and ra for specific year
    if balrog_year == "Y1":
        start = -60
        stop = -40
    elif balrog_year == "Y3":
        start = -40
        stop = -3
    elif balrog_year == "ALL":
        start = -60
        stop = -3
    else:
        raise Exception("Wrong Balrog Year")

    # Select region for specific year of balrog
    selector_of_region = df["DEC"].between(start, stop, inclusive=False)
    df_specific_region = df[selector_of_region]

    # Calculate the fraction to get only the needed number of training data. If length_of_data is -1. Than all data will
    # be used
    fraction = 1
    if not length_of_data == -1:
        fraction = length_of_data/len(df_specific_region)
    pd_reduced_data = df_specific_region.sample(frac=fraction)

    # Creating a new pandas Dataframe with a random number of data and save in current folder
    lst_deep_field_data = list(pd_reduced_data["BDF_FLUX_DERED_R"])
    lst_flux_data = list(pd_reduced_data["unsheared/flux_r"])
    lst_df_and_flux = []
    for idx, item in enumerate(lst_deep_field_data):
        lst_df_and_flux.append([item, lst_flux_data[idx]])
    pd_deep_field_and_flux = pd.DataFrame(lst_df_and_flux, columns=["deep field", "flux"])
    # pd_deep_field_and_flux.to_pickle(f"deep_balrog_{len(pd_deep_field_and_flux)}_Year_{balrog_year}.pkl")


if __name__ == "__main__":
    load_simulated_data(
        filename="deep_ugriz.mof02_sn.jhk.ff04_c.jhk.ff02_052020_realerrors_May20calib.pkl",
        balrog_year="ALL",
        length_of_data=40000)
