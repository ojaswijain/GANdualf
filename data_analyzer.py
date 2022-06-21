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


def generate_normal_distribution(size, mu, sigma, num=1, as_tensor=True):
    """
    Generate uniform distributed random data for discriminator.

    Args:
        size: size of the tensor

    Returns:
        random data as torch tensor
    """
    # random_data = torch.randn(size)

    if as_tensor is False:
        return np.random.normal(mu, sigma, size=(size, num))
    return torch.FloatTensor([np.random.normal(mu, sigma, size=(size, num))[0][0]])


def generate_uniform_distribution(size, low, high, num=1, as_tensor=True):
    """
    Generate normal distributed random data for generator.

    Args:
        size: size of the tensor

    Returns:
        random data as torch tensor
    """
    # random_data = torch.rand(size)

    if not as_tensor:
        return np.random.uniform(low, high, size=(size, num))
    return torch.FloatTensor([np.random.uniform(low, high, size=(size, num))[0][0]])


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
    pd_noise = df["flux"] - df["deep field"] * 1.12

    # calculate the standard scale of deep field to avoid huge numbers
    pd_standardized_deep_field = standard_scaler(pd_deep_field_data)
    pd_standardized_flux = standard_scaler(pd_flux_data)
    pd_standardized_noise = standard_scaler(pd_noise)

    plt.scatter(pd_standardized_deep_field, pd_standardized_noise, s=1)
    plt.xlim(-2, 2)
    plt.ylim(-4, 4)
    plt.show()

    # create train and test arrays
    pd_train_deep_field = pd_standardized_deep_field[:int(len(pd_standardized_deep_field) * .8)]
    pd_train_flux = pd_standardized_flux[:int(len(pd_standardized_flux) * .8)]
    pd_train_noise = pd_standardized_noise[:int(len(pd_standardized_noise) * .8)]

    plt.scatter(pd_train_deep_field, pd_train_noise, s=1)
    plt.xlim(-2, 2)
    plt.ylim(-4, 4)
    plt.show()

    pd_test_deep_field = pd_standardized_deep_field[int(len(pd_standardized_deep_field) * .8):]
    pd_test_flux = pd_standardized_flux[int(len(pd_standardized_flux) * .8):]
    pd_test_noise = pd_standardized_noise[int(len(pd_standardized_noise) * .8):]

    plt.scatter(pd_test_deep_field, pd_test_noise, s=1)
    plt.xlim(-2, 2)
    plt.ylim(-4, 4)
    plt.show()

    df = pd.DataFrame({"noise": pd_standardized_noise, "train noise": pd_train_noise, "test noise": pd_test_noise})
    sns.histplot(df)
    plt.show()

    exit()

    lst_gandalf_training = []
    for idx, deep_field_value in enumerate(pd_train_deep_field):
        lst_gandalf_training.append({
            "deep field value scaled": deep_field_value,
            "deep field value": inversion(pd_deep_field_data, deep_field_value),
            "flux value scaled": pd_train_flux[idx],
            "flux value": inversion(pd_flux_data, pd_train_flux[idx]),
            "noise scaled": pd_train_noise[idx],
            "noise": inversion(pd_noise, pd_train_noise[idx])
        })

    lst_gandalf_test = []
    for idx, deep_field_value in enumerate(pd_test_deep_field):
        lst_gandalf_test.append({
            "deep field value scaled": deep_field_value,
            "deep field value": inversion(pd_deep_field_data, deep_field_value),
            "flux value scaled": pd_test_flux[idx+int(len(pd_standardized_flux) * .8)],
            "flux value": inversion(pd_flux_data, pd_test_flux[idx+int(len(pd_standardized_flux) * .8)]),
            "noise scaled": pd_test_noise[idx+int(len(pd_standardized_noise)*.8)],
            "noise": inversion(pd_test_noise, pd_test_noise[idx+int(len(pd_standardized_noise)*.8)])
        })

    return {
        "training data": lst_gandalf_training,
        "test data": lst_gandalf_test,
        "deep field data": pd_deep_field_data,
        "flux data": pd_flux_data,
        "noise data": pd_noise
    }


class Hobbit(nn.Module):

    def __init__(self, lr):
        """
        Constructor

        Args:
            lr: float - learning rate
        """

        # Constructor of parent class
        super().__init__()

        # Define the layer of the neural net
        self.model = nn.Sequential(
            nn.Linear(1+1, 40),     # Linear layer 2 -> 40
            nn.LeakyReLU(0.02),     # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(40, 600),     # Linear layer 40 -> 600
            nn.LeakyReLU(0.02),     # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(600, 400),    # Linear layer 600 -> 400
            nn.LeakyReLU(0.02),     # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.LayerNorm(400),      # Normalize the values in the neural net to avoid large values

            nn.Linear(400, 1),      # Linear layer 400 -> 1
            nn.Sigmoid()            # Activation function (Sigmoid with Values between [0,1])
        )

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = nn.MSELoss()

        # Optimizer to calculate the weight changes
        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)

        # Define variable for progress list and counts
        self.counter = 0
        self.lst_progress = []

    def forward(self, input_tensor, label_tensor):
        """
        Forward function of the neural net

        Args:
            input_tensor: torch tensor of the input value
            label_tensor: torch tensor of the label value

        Returns:
            output of the defined neural net after calculation with one input value
        """

        # combine the input tensor and the label tensor to one input tensor
        inputs = torch.cat((input_tensor, label_tensor))
        return self.model(inputs)

    def train(self, inputs, label_tensor, targets):
        """
        Training method of the discriminator

        Args:
            inputs: torch tensor of the input value
            label_tensor: torch tensor of the label value
            targets: torch tensor of the target value
            data_type: type of data
        """
        # Calculate the output of the neural net
        outputs = self.forward(inputs, label_tensor)

        # Calculate the loss of the neural net
        loss = self.loss_function(outputs, targets)

        # Update the counter and calculate the progress after 10 counts and write output after 10000 counts
        self.counter += 1
        if self.counter % 10 == 0:
            self.lst_progress.append(loss.item())
        if self.counter % 10000 == 0:
            print("counter = %i" % self.counter)

        # Set optimizer to zero
        self.optimiser.zero_grad()

        # Perform backpropagation and update the weights of the neural net
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def plot_progress(self, plot_name=None, show=True):
        """
        Plotting method to show the progress

        Args:
            plot_name: path and name of the plot image. If None, no plot image will be created. Default value is None
            show: If True, the plot image will be shown. Default value is True
        """

        plt.close("all")

        # Create dataframe of progress list
        df_loss = pd.DataFrame({"loss": self.lst_progress})

        # Create plot
        df_ax = df_loss.plot(
            ylim=(0),
            figsize=(16, 8),
            alpha=0.1,
            marker=".",
            grid=True,
            yticks=(0, 0.25,  0.5, 0.693, 1.0, 5.0))

        # Get figure of current plot
        df_fig = df_ax.get_figure()

        # If shown==True, plot image will be shown. Else pass
        if show:
            df_fig.show()

        # If plot image is not None. The plot image will be saved with given path. Else pass
        if plot_name is not None:
            df_fig.savefig(plot_name)

        # Clear and close open figure to avoid memory overload
        df_fig.clf()
        plt.close(df_fig)


def hobbit_training(filename, lr, epochs=1, plotting_progress=True, save_hobbit=True):
    """
    Function to train the GAN

    Args:
        filename: Path of the *.pkl-file with the training data
        lr: learning rate in float
        epochs: maximum epochs, Integer
        plotting_progress: True to plot the progress

    Returns: the trained discriminator and generator
    """

    # Set output paths
    path_gan_training_learning_rate = f"Output/Hobbit-Training/learning_rate_{lr}"
    path_gan_training_output_balrog = f"Output/Hobbit-Training/learning_rate_{lr}/balrog_progress_output"
    path_gan_training_output_gandalf = f"Output/Hobbit-Training/learning_rate_{lr}/gandalf_progress_output"

    # Init the generator (balrog) and the discriminator (gandalf)
    hobbit = Hobbit(lr=lr)

    # Load the data
    dict_loaded_data = load_simulated_data(filename=filename)

    # Get the actual system time to calculate the duration from beginning
    start_time = datetime.now()
    # Start training
    for epoch in range(epochs):
        print(
            f"######################################  epoch {epoch + 1}  #############################################")

        # Get the actual system time to calculate the duration for each epoch
        epoch_time = datetime.now()

        # Iterate over training data
        for idx, training_item in enumerate(dict_loaded_data["training data"]):
            # train discriminator to true values
            hobbit_loss_real_data = hobbit.train(
                generate_uniform_distribution(low=-2, high=2, size=1),
                torch.FloatTensor([training_item["deep field value scaled"]]),
                torch.FloatTensor([training_item["noise scaled"]])
            )

            # Give progress update every 100 training elements
            if idx % 100 == 0:
                print(f"Status {idx / len(dict_loaded_data['training data']) * 100}% \t "
                      f"gandalf loss with real data: {hobbit_loss_real_data}; \t ")

        # Calculate the time difference
        delta_start_time = datetime.now() - start_time
        delta_epoch_time = datetime.now() - epoch_time
        print("Elapsed time during since start in seconds:", delta_start_time.seconds)
        print("Elapsed time during for epoch in seconds:", delta_epoch_time.seconds)

        # Plotting the progress if True
        if plotting_progress:
            # Create folder if they doesn't exist
            if not os.path.exists("Output/Hobbit-Training"):
                os.mkdir("Output/Hobbit-Training")
            if not os.path.exists(path_gan_training_learning_rate):
                os.mkdir(path_gan_training_learning_rate)
            if not os.path.exists(path_gan_training_output_balrog):
                os.mkdir(path_gan_training_output_balrog)
            if not os.path.exists(path_gan_training_output_gandalf):
                os.mkdir(path_gan_training_output_gandalf)

            # Plotting the progress
            hobbit.plot_progress(
                plot_name=f"{path_gan_training_output_gandalf}/hobbit_learning_progress_epoch_{epoch + 1}.png",
                show=False)

        if save_hobbit:
            torch.save(hobbit, f"save_nn/hobbit_{lr}.pt")

    return hobbit


def hobbit_test(filename, lr, hobbit, plotting_result=True, plotting_standard_deviation=False):
    """
    Function to test the gan with Test values and to verify how high the standard deviation is.

    Args:
        filename: Path of the *.pkl-file with the training data
        lr: learning rate in float
        balrog: Trained balrog generator
        plotting_result: True to plot the result
        plotting_standard_deviation: True to plot the standard deviation

    Returns:
    """

    # Load the data
    dict_loaded_data = load_simulated_data(filename=filename)

    # Init new lists
    lst_deep_field = []
    lst_noise = []
    lst_result = []
    lst_standard_deviation = []
    lst_power_diff = []
    # for _ in range(1000):
    for idx, test_item in enumerate(dict_loaded_data["test data"]):
        result = hobbit.forward(generate_uniform_distribution(low=-2, high=2, size=1), torch.FloatTensor([test_item["deep field value scaled"]]))
        lst_result.append(inversion(dict_loaded_data["noise data"], result[0].item()))
        lst_deep_field.append(test_item["deep field value"])
        lst_noise.append(test_item["noise"])
        lst_power_diff.append((result[0].item() - test_item["noise scaled"])**2)

    s_deviation = np.sqrt(np.sum(lst_power_diff)/len(lst_power_diff))

    # Calculate accuracy and the mean of the results
    accuracy = 100 * (1 - s_deviation)

    # Print results of the test
    print("accuracy", accuracy)

    # Creating a dictionary and pandas data frame with the results
    dict_result = {'result': lst_result, 'noise': lst_noise, "deep field": lst_deep_field}
    df = pd.DataFrame(dict_result)

    # Plotting the results if True
    if plotting_result:
        # Create folder if they doesn't exist
        if not os.path.exists("Output/Hobbit-Test"):
            os.mkdir("Output/Hobbit-Test")

        # Plot the deep field against flux and against result
        plt.plot(df["deep field"], df["noise"], ".b", label="Soll")
        plt.plot(df["deep field"], df["result"], ".r", label="Ist")
        plt.title(f"Result {lr}; Accuracy {accuracy}")
        plt.xlabel("Flux Deep Field")
        plt.ylabel("Noise")
        plt.savefig(f"Output/Hobbit-Test/result_lr_{lr}_accuracy_{accuracy}.png")
        plt.show()
        plt.clf()
        plt.close()

    sns.histplot(df["result"])
    plt.savefig(f"Output/Hobbit-Test/histplot_result_lr_{lr}.png")
    plt.show()
    sns.histplot(df["noise"])
    plt.savefig(f"Output/Hobbit-Test/histplot_noise_lr_{lr}.png")
    plt.show()


def main(epochs, filename, lst_learning_rates, load_hobbit, save_hobbit, train_hobbit, test_hobbit, run_test_hobbit):
    """"""

    if train_hobbit is True:
        for lr in lst_learning_rates:
            model_hobbit = hobbit_training(filename=filename, lr=lr, epochs=epochs, save_hobbit=save_hobbit)

            if run_test_hobbit:
                hobbit_test(
                    filename=filename,
                    lr=lr,
                    hobbit=model_hobbit,
                    plotting_result=True)

    exit()


if __name__ == "__main__":
    lst_lr = []
    for i in range(1, 9):
        lst_lr.append(1/(10**i))
    main(
        epochs=100,
        filename=r"../../Data/simulated_data_linear_gaussian_1000.pkl",
        lst_learning_rates=lst_lr,
        load_hobbit=False,
        save_hobbit=False,
        train_hobbit=True,
        test_hobbit=False,
        run_test_hobbit=True
    )
