from src.Discriminator.discriminator import GandalfDisc
from src.Generator.generator import BalrogGen
import pickle
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
import seaborn as sns
from decimal import Decimal
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def flux2mag(flux, zero_pt=30):
    # convert flux to magnitude
    return zero_pt - 2.5 * np.log10(flux)


def load_data(filepath):
    """"""
    # open file
    infile = open(filepath, 'rb')  # filename

    # load pickle as pandas dataframe
    df = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()
    return df


def plot_balrog(data_frame):
    """"""
    measured_mag = flux2mag(data_frame["unsheared/flux_r"])
    plt.scatter(data_frame["true_bdf_mag_deredden_r"],  measured_mag, s=1)
    plt.show()


def main(filepath, col):
    """"""
    df_balrog = load_data(filepath=filepath)
    plot_balrog(df_balrog)
    for col in df_balrog:
        print(col)

        lst_values = []
        for br in df_balrog[col]:
            if br in lst_values:
                continue
            lst_values.append(br)
            if len(lst_values) == 10:
                lst_values = []
                lst_values.append(df_balrog[col].min())
                lst_values.append(df_balrog[col].max())
                break

        print(lst_values)


if __name__ == "__main__":
    lst_lr = []
    for i in range(1, 9):
        lst_lr.append(1/(10**i))

    main(
        filepath=f"..\..\data\data.pkl",
        col="unsheared/R11"
    )