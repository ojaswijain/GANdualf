from helper_functions import concatenate_lists
import torch
import torch.nn as nn
import numpy as np


class Discriminator2(nn.Module):

    def __init__(self, lr):
        """
        Constructor

        Args:
            lr: float - learning rate
        """

        # Constructor of parent class
        super().__init__()
        self.learning_rate = lr

        # Define the layer of the neural net
        self.model = nn.Sequential(
            nn.Linear(1, 40),  # Linear layer 2 -> 128
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(40, 400),  # Linear layer 128 -> 128
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            #nn.Linear(400, 600),  # Linear layer 128 -> 128
            #nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(400, 400),  # Linear layer 128 -> 128
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.LayerNorm(400),  # Normalize the values in the neural net to avoid large values

            nn.Linear(400, 1),  # Linear layer 128 -> 128
            nn.Sigmoid()  # Activation function (Sigmoid with Values between [0,1])
        )

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = nn.BCELoss()

        # Optimizer to calculate the weight changes
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define variable for progress list and counts
        self.counter = 0
        self.lst_progress = []

    def forward(self, input_tensor):
        """
        Forward function of the neural net

        Args:
            input_tensor: torch tensor of the input value

        Returns:
            output of the defined neural net after calculation with one input value
        """

        return self.model(input_tensor[:,:1])

    def train_on_batch(self, inputs, labels, targets, batch_size=64):
        """
        Training method of the discriminator

        Args:
            inputs: numpy array or list of input value
            labels: numpy array or list of label value
            targets: numpy array or list of target value
            batch_size: size of batch
        """
        if type(inputs) is list:
            inputs = concatenate_lists(inputs)

        if type(labels) is list:
            labels = concatenate_lists(labels)

        if type(targets) is list:
            targets = concatenate_lists(targets)

        stacked_inputs = np.column_stack((inputs, labels))

        batch_counter = 0
        while batch_counter < len(stacked_inputs):
            rnd_indices = np.random.choice(len(stacked_inputs), size=batch_size, replace=False)
            batch_data = stacked_inputs[rnd_indices]
            batch_targets = targets[rnd_indices]

            # Calculate the output of the neural net
            outputs = self.forward(torch.FloatTensor(batch_data))

            # Calculate the loss of the neural net
            loss = self.loss_function(outputs, torch.FloatTensor(batch_targets))

            # Update the counter and calculate the progress after 10 counts and write output after 10000 counts
            self.lst_progress.append(loss.item())

            # Set optimizer to zero
            self.optimiser.zero_grad()

            # Perform backpropagation and update the weights of the neural net
            loss.backward()
            self.optimiser.step()
            batch_counter += batch_size
        return self.lst_progress

if __name__ == "__main__":
    print(1)
