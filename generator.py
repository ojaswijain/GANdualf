from helper_functions import concatenate_lists
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, discriminator, lr, input_dim):
        """
                Constructor

                Args:
                    lr: float - learning rate
                """

        # Constructor of parent class
        super().__init__()
        self.learning_rate = lr
        self.input_dim = input_dim
        self.discriminator = discriminator
        #self.discriminator2 = discriminator2

        # Define the layer of the neural net

        self.model = nn.Sequential(
            nn.Linear(input_dim, 400),  # Linear layer 2 -> 400
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(400, 600),  # Linear layer 400 -> 600
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            #nn.Linear(600, 600),  # Linear layer 400 -> 600
            #nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than

            nn.Linear(600, 400),  # Linear layer 400 -> 600
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)

            nn.Linear(400, 40),  # Linear layer 400 -> 600
            nn.LeakyReLU(0.02),  # Activation function (Rectified Linear Unit with left values smaller than 0)
            nn.LayerNorm(40),

            nn.Linear(40, 2),  # Linear layer 40 -> 2
            nn.LeakyReLU(2)  # Activation function (Rectified Linear Unit with left values smaller than 0)
        )

        # Optimizer to calculate the weight changes
        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = self.discriminator.loss_function

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

        return self.model(input_tensor)

    def train_on_batch(self, inputs, labels, batch_size=64):
        """
        Training method of the discriminator

        Args:
            inputs: torch tensor of the input value
            labels: torch tensor of the label value
            targets: torch tensor of the target value
            batch_size: size of batch
        """

        if type(inputs) is list:
            inputs = concatenate_lists(inputs)

        if type(labels) is list:
            labels = concatenate_lists(labels)

        batch_counter = 0
        while batch_counter < len(inputs):
            rnd_indices = np.random.choice(len(inputs), size=batch_size, replace=False)
            batch_data = inputs[rnd_indices]
            batch_labels = labels[rnd_indices]

            # Calculate the output of the generator neural net
            generator_output = self.forward(torch.FloatTensor(batch_data))
            generator_output = generator_output.reshape((batch_size,2))

            #cat_generator_output = torch.column_stack((generator_output, torch.FloatTensor(batch_labels)))
            cat_generator_output = generator_output
            discriminator_output = self.discriminator.forward(cat_generator_output)

            # Calculate the loss of the neural net
            loss = (self.loss_function(discriminator_output, torch.FloatTensor(np.ones((batch_size, 1)))))

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

