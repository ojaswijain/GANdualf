from helper_functions import calc_ks_critical_value, reverses_standard_scaler, concatenate_lists, standard_deviation
from data_loader import load_simulated_data
from generator import Generator
from discriminator import Discriminator
from discriminator2 import Discriminator2
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kstest


class GenerativeAdversarialNetwork(nn.Module):

    def __init__(self, lr_g, lr_d, input_dim):
        """"""
        # Constructor of parent class
        super().__init__()
        # np.random.seed(123)
        self.learning_rate_generator = lr_g
        self.learning_rate_discriminator = lr_d
        self.input_dim = input_dim

        self.D = Discriminator(lr=self.learning_rate_discriminator)
        self.D2 = Discriminator2(lr=self.learning_rate_discriminator * 1)
        self.G = Generator(discriminator=self.D, lr=self.learning_rate_generator, input_dim=input_dim)

        # Define variable for progress list and counts
        self.counter = 0
        self.lst_loss_discriminator_fake = []
        self.lst_loss_epoch_discriminator_fake = []
        self.lst_mean_loss_per_epoch_discriminator_fake = []

        self.lst_loss_discriminator_true = []
        self.lst_loss_epoch_discriminator_true = []
        self.lst_mean_loss_per_epoch_discriminator_true = []

        self.lst_loss_discriminator2_fake = []
        self.lst_loss_epoch_discriminator2_fake = []
        self.lst_mean_loss_per_epoch_discriminator2_fake = []

        self.lst_loss_discriminator2_true = []
        self.lst_loss_epoch_discriminator2_true = []
        self.lst_mean_loss_per_epoch_discriminator2_true = []

        self.lst_loss_generator = []
        self.lst_loss_epoch_generator = []
        self.lst_mean_loss_per_epoch_generator = []

        self.lst_kstest = []
        self.lst_critical_values = []
        self.lst_epochs = []

        self.lst_mean_diff = []
        self.lst_stddev_diff = []

    def train_on_batch(self, inputs_discriminator, labels_discriminator, inputs_generator,
                       labels_generator, target_path, epoch, batch_size=64):
        """
        Training method of the discriminator

        Args:
            discriminator: discriminator
            inputs: torch tensor of the input value
            labels: torch tensor of the label value
            targets: torch tensor of the target value
            batch_size: size of batch
        """

        if type(inputs_discriminator) is list:
            inputs_discriminator = concatenate_lists(inputs_discriminator)

        if type(labels_discriminator) is list:
            labels_discriminator = concatenate_lists(labels_discriminator)

        if type(inputs_generator) is list:
            inputs_generator = concatenate_lists(inputs_generator)

        if type(labels_generator) is list:
            labels_generator = concatenate_lists(labels_generator)

        # stacked_inputs_discriminator = np.column_stack((inputs_discriminator, labels_discriminator))
        # stacked_inputs_generator = np.column_stack((inputs_generator, labels_generator))

        batch_counter = 0
        while batch_counter < len(inputs_generator):
            # Train Discriminator #############################################################################
            d_random_real = np.random.choice(len(inputs_discriminator), size=batch_size, replace=False)
            lst_loss_discriminator_true = self.D.train_on_batch(
                inputs=[inputs_discriminator[d_random_real]],
                labels=[labels_discriminator[d_random_real]],
                targets=[np.ones((batch_size, 1))],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_discriminator_true.append(lst_loss_discriminator_true[-1])
                self.lst_loss_epoch_discriminator_true.append(lst_loss_discriminator_true[-1])

            outputs_generator_4_D = self.G.forward(torch.FloatTensor(inputs_generator)).detach().numpy()
            outputs_generator_4_D = outputs_generator_4_D.reshape((len(inputs_generator),2))
            d_random_fake = np.random.choice(len(outputs_generator_4_D), size=batch_size, replace=False)
            lst_loss_discriminator_fake = self.D.train_on_batch(
                inputs=[outputs_generator_4_D[d_random_fake][:, 0:1]],
                labels=[outputs_generator_4_D[d_random_fake][:, 1:2]],
                targets=[np.zeros((batch_size, 1))],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_discriminator_fake.append(lst_loss_discriminator_fake[-1])
                self.lst_loss_epoch_discriminator_fake.append(lst_loss_discriminator_fake[-1])

            # Train Generator #####################################################################################

            g_random = np.random.choice(len(inputs_discriminator), size=batch_size, replace=False)
            lst_loss_generator = self.G.train_on_batch(
                inputs=[inputs_generator[g_random]],
                labels=[labels_generator[g_random]],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_generator.append(lst_loss_generator[-1])
                self.lst_loss_epoch_generator.append(lst_loss_generator[-1])

            batch_counter += batch_size

        deviation_from_equilibrium_discriminator_true = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator_true))
        deviation_from_equilibrium_discriminator_fake = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator_fake))
        deviation_from_equilibrium_generator = np.abs(0.693 - np.mean(self.lst_loss_epoch_generator))
        return deviation_from_equilibrium_discriminator_true, deviation_from_equilibrium_discriminator_fake, \
               deviation_from_equilibrium_generator

    def train_on_batch2(self, inputs_discriminator, labels_discriminator, inputs_generator,
                       labels_generator, target_path, epoch, batch_size=64):
        """
        Training method of the discriminator

        Args:
            discriminator: discriminator
            inputs: torch tensor of the input value
            labels: torch tensor of the label value
            targets: torch tensor of the target value
            batch_size: size of batch
        """

        if type(inputs_discriminator) is list:
            inputs_discriminator = concatenate_lists(inputs_discriminator)

        if type(labels_discriminator) is list:
            labels_discriminator = concatenate_lists(labels_discriminator)

        if type(inputs_generator) is list:
            inputs_generator = concatenate_lists(inputs_generator)

        if type(labels_generator) is list:
            labels_generator = concatenate_lists(labels_generator)

        # stacked_inputs_discriminator = np.column_stack((inputs_discriminator, labels_discriminator))
        stacked_inputs_generator = np.column_stack((inputs_generator, labels_generator))

        batch_counter = 0
        while batch_counter < len(stacked_inputs_generator):
            # Train Discriminator #############################################################################
            d_random_real = np.random.choice(len(inputs_discriminator), size=batch_size, replace=False)
            lst_loss_discriminator2_true = self.D2.train_on_batch(
                inputs=[inputs_discriminator[d_random_real]],
                labels=[labels_discriminator[d_random_real]],
                targets=[np.ones((batch_size, 1))],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_discriminator2_true.append(lst_loss_discriminator2_true[-1])
                self.lst_loss_epoch_discriminator2_true.append(lst_loss_discriminator2_true[-1])

            outputs_generator_4_D = self.G.forward(torch.FloatTensor(inputs_generator)).detach().numpy()
            outputs_generator_4_D = outputs_generator_4_D.reshape((len(inputs_generator),2))
            d_random_fake = np.random.choice(len(outputs_generator_4_D), size=batch_size, replace=False)
            lst_loss_discriminator2_fake = self.D2.train_on_batch(
                inputs=[outputs_generator_4_D[d_random_fake][:,0:1]],
                labels=[outputs_generator_4_D[d_random_fake][:,1:2]],
                targets=[np.zeros((batch_size, 1))],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_discriminator2_fake.append(lst_loss_discriminator2_fake[-1])
                self.lst_loss_epoch_discriminator2_fake.append(lst_loss_discriminator2_fake[-1])

            # Train Generator #####################################################################################

            g_random = np.random.choice(len(inputs_discriminator), size=batch_size, replace=False)
            lst_loss_generator = self.G.train_on_batch(
                inputs=[inputs_generator[g_random]],
                labels=[labels_generator[g_random]],
                batch_size=batch_size
            )

            self.counter += 1
            if self.counter % 1 == 0:
                self.lst_loss_generator.append(lst_loss_generator[-1])
                self.lst_loss_epoch_generator.append(lst_loss_generator[-1])

            batch_counter += batch_size

        deviation_from_equilibrium_discriminator2_true = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator2_true))
        deviation_from_equilibrium_discriminator2_fake = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator2_fake))
        deviation_from_equilibrium_generator = np.abs(0.693 - np.mean(self.lst_loss_epoch_generator))
        return deviation_from_equilibrium_discriminator2_true, deviation_from_equilibrium_discriminator2_fake, \
               deviation_from_equilibrium_generator

    # def plot_statistical_test(self, test_data, lst_gandalf_df_flux_scaled, lst_gandalf_wf_flux_scaled,
    #                           plot_name_statistical, plot_name_compare, batch_size, compare_plot=False, show=True,
    #                           epoch=10):
    #     """
    #     Plotting method to show the progress
    #
    #     Args:
    #         plot_name: path and name of the plot image. If None, no plot image will be created. Default value is None
    #         show: If True, the plot image will be shown. Default value is True
    #     """
    #     self.lst_epochs.append(epoch)
    #     sns.set_theme()
    #
    #     plt.close("all")
    #
    #     # Create dataframe of progress list
    #     df_loss = pd.DataFrame({"discriminator fake data": self.lst_loss_discriminator_fake,
    #                             "discriminator real data": self.lst_loss_discriminator_true,
    #                             "discriminator2 fake data": self.lst_loss_discriminator2_fake,
    #                             "discriminator2 real data": self.lst_loss_discriminator2_true,
    #                             "generator": self.lst_loss_generator})
    #
    #     statistical_figure, ((stat_ax1, stat_ax2, stat_ax3), (stat_ax4, stat_ax5, stat_ax6)) = plt.subplots(nrows=2,
    #                                                                                                         ncols=3)
    #     statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    #     statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)
    #
    #     # Create plot
    #     df_loss.plot(
    #         figsize=(16, 9),
    #         ylim=(0),
    #         alpha=0.5,
    #         marker=".",
    #         grid=True,
    #         yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
    #         ax=stat_ax1)
    #
    #     stat_ax1.set_xlabel("counts -> loss per batch", fontsize=10, loc='right')
    #     stat_ax1.set_ylabel("loss, Sum[-y*ln[x]]", fontsize=10, loc='top')
    #     stat_ax1.set_title(f"Binary-Cross-Entropy -> equilibrium at 0.693")
    #
    #     dev_loss_discriminator_true = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator_true))
    #     dev_loss_discriminator_fake = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator_fake))
    #     dev_loss_discriminator2_true = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator2_true))
    #     dev_loss_discriminator2_fake = np.abs(0.693 - np.mean(self.lst_loss_epoch_discriminator2_fake))
    #     dev_loss_generator = np.abs(0.693 - np.mean(self.lst_loss_epoch_generator))
    #
    #     self.lst_mean_loss_per_epoch_discriminator_true.append(dev_loss_discriminator_true)
    #     self.lst_mean_loss_per_epoch_discriminator_fake.append(dev_loss_discriminator_fake)
    #     self.lst_mean_loss_per_epoch_discriminator2_true.append(dev_loss_discriminator2_true)
    #     self.lst_mean_loss_per_epoch_discriminator2_fake.append(dev_loss_discriminator2_fake)
    #     self.lst_mean_loss_per_epoch_generator.append(dev_loss_generator)
    #
    #     # Plot mean loss per Batch
    #     stat_ax2.plot(
    #         self.lst_epochs,
    #         self.lst_mean_loss_per_epoch_discriminator_fake,
    #         marker=".",
    #         alpha=0.5,
    #         label=f"deviation discriminator fake data = {dev_loss_discriminator_fake:3.4f}")
    #     stat_ax2.plot(
    #         self.lst_epochs,
    #         self.lst_mean_loss_per_epoch_discriminator_true,
    #         marker=".",
    #         alpha=0.5,
    #         label=f"deviation discriminator real data = {dev_loss_discriminator_true:3.4f}")
    #     stat_ax2.plot(
    #         self.lst_epochs,
    #         self.lst_mean_loss_per_epoch_discriminator2_fake,
    #         marker=".",
    #         alpha=0.5,
    #         label=f"deviation discriminator-2 fake data = {dev_loss_discriminator2_fake:3.4f}")
    #     stat_ax2.plot(
    #         self.lst_epochs,
    #         self.lst_mean_loss_per_epoch_discriminator2_true,
    #         marker=".",
    #         alpha=0.5,
    #         label=f"deviation discriminator-2 real data = {dev_loss_discriminator2_true:3.4f}")
    #     stat_ax2.plot(
    #         self.lst_epochs,
    #         self.lst_mean_loss_per_epoch_generator,
    #         marker=".",
    #         alpha=0.5,
    #         label=f"deviation generator = {dev_loss_generator:3.4f}")
    #     stat_ax2.set_xlabel("Epoch", fontsize=10, loc='right')
    #     stat_ax2.set_ylabel("|0.693 - Sum_i[Sum_j[-y*ln[x_j]]_i]/n|", fontsize=10, loc='top')
    #     stat_ax2.set_title(f"Deviation from equilibrium")
    #     stat_ax2.legend()
    #
    #     deep_field = reverses_standard_scaler(
    #         arr=test_data["deep field scaled"],
    #         mean=test_data["deep field mean"],
    #         std_dev=test_data["deep field standard deviation"])
    #     wide_field = reverses_standard_scaler(
    #         arr=test_data["wide field scaled"],
    #         mean=test_data["wide field mean"],
    #         std_dev=test_data["wide field standard deviation"])
    #     deep_field_generated = reverses_standard_scaler(
    #         arr=np.array(lst_gandalf_df_flux_scaled),
    #         mean=test_data["deep field mean"],
    #         std_dev=test_data["deep field standard deviation"])
    #     wide_field_generated = reverses_standard_scaler(
    #         arr=np.array(lst_gandalf_wf_flux_scaled),
    #         mean=test_data["wide field mean"],
    #         std_dev=test_data["wide field standard deviation"])
    #     stat_ax3.plot(
    #         deep_field,
    #         wide_field,
    #         ".",
    #         color="#DD8452",
    #         alpha=0.5,
    #         label="wide field flux")
    #     stat_ax3.plot(
    #         deep_field_generated,
    #         wide_field_generated,
    #         ".",
    #         color="#55A868",
    #         alpha=0.5,
    #         label="generated wide field flux")
    #     stat_ax3.legend()
    #     # stat_ax3.set_ylim((18000, 48000))
    #     # stat_ax3.set_xlim((18000, 48000))
    #     stat_ax3.set_xlabel("Deep field flux", fontsize=10, loc='right')
    #     stat_ax3.set_ylabel("Wide field flux", fontsize=10, loc='top')
    #     stat_ax3.set_title(f"Test data result")
    #
    #     df_distribution_wf_flux = pd.DataFrame({
    #         "generated wide field flux": wide_field_generated,
    #         "wide field flux": wide_field
    #     })
    #     sns.histplot(df_distribution_wf_flux, stat="probability", kde=True, kde_kws={"bw_adjust": 3},
    #                  ax=stat_ax4, palette=["#55A868", "#DD8452"])
    #     stat_ax4.set_xlabel("Wide field flux", fontsize=10, loc='right')
    #     stat_ax4.set_title(f"Probability of wide field fluxes")
    #
    #     cdf_wide_field_generated = np.array(ECDF(wide_field_generated).x)
    #     cdf_wide_field_generated = cdf_wide_field_generated[np.isfinite(cdf_wide_field_generated)]
    #     cdf_wide_field = np.array(ECDF(wide_field).x)
    #     cdf_wide_field = cdf_wide_field[np.isfinite(cdf_wide_field)]
    #
    #     df_cdf = pd.DataFrame({
    #         "difference of cdf": np.abs(cdf_wide_field_generated - cdf_wide_field) + np.mean(cdf_wide_field),
    #         "cdf wide field": cdf_wide_field,
    #         "cdf wide field generated": cdf_wide_field_generated,
    #     })
    #     df_cdf.plot(
    #         grid=True,
    #         ax=stat_ax5)
    #     stat_ax5.set_xlabel("ID", fontsize=10, loc='right')
    #     stat_ax5.set_ylabel("Wide field flux", fontsize=10, loc='top')
    #     stat_ax5.set_title(f"Cumulative distribution functions")
    #
    #     kstest_stat = kstest(
    #         rvs=wide_field,
    #         cdf=wide_field_generated
    #     )
    #     self.lst_critical_values.append(calc_ks_critical_value(len(wide_field), len(wide_field_generated), a=0.1))
    #     self.lst_kstest.append(kstest_stat[0])
    #
    #     stat_ax6.plot(
    #         self.lst_epochs,
    #         self.lst_kstest,
    #         alpha=0.5,
    #         marker=".",
    #         label=f"Kolmogorov–Smirnov value: {self.lst_kstest[-1]:5.4f}")
    #     stat_ax6.plot(
    #         self.lst_epochs,
    #         self.lst_critical_values,
    #         alpha=0.5,
    #         marker=".",
    #         label=f"Kolmogorov–Smirnov critical value: {self.lst_critical_values[-1]:5.4f}")
    #     stat_ax6.set_xlabel("Epoch", fontsize=10, loc='right')
    #     stat_ax6.set_ylabel("Kolmogorov–Smirnov value", fontsize=10, loc='top')
    #     stat_ax6.set_title("Kolmogorov–Smirnov test")
    #     stat_ax6.legend()
    #
    #     # If shown==True, plot image will be shown. Else pass
    #     # show = True
    #     if show is True:
    #         plt.show()
    #         exit()
    #
    #     # If plot image is not None. The plot image will be saved with given path. Else pass
    #     if plot_name_statistical is not None:
    #         statistical_figure.savefig(plot_name_statistical, dpi=200)
    #
    #     if compare_plot is True and plot_name_compare is not None:
    #         statistical_figure.suptitle(f"Epochs: {epoch}; "
    #                                     f"Input Tensor {self.input_dim}; "
    #                                     f"Learning rate discriminator {self.learning_rate_discriminator}; "
    #                                     f"Learning rate generator {self.learning_rate_generator}; "
    #                                     f"Batch size {batch_size}", fontsize=16)
    #         statistical_figure.savefig(plot_name_compare, dpi=200)
    #
    #     # Clear and close open figure to avoid memory overload
    #     statistical_figure.clf()
    #     plt.close(statistical_figure)
    #
    #     lst_data_types = []
    #     lst_x_values = []
    #     lst_y_values = []
    #     for idx, item in enumerate(deep_field_generated):
    #         lst_data_types.append("generated data")
    #         lst_x_values.append(item)
    #         lst_y_values.append(wide_field_generated[idx])
    #
    #         lst_data_types.append("real data")
    #         lst_y_values.append(wide_field[idx])
    #         lst_x_values.append(deep_field[idx])
    #
    #     self.lst_loss_epoch_discriminator_true = []
    #     self.lst_loss_epoch_discriminator_fake = []
    #     self.lst_loss_epoch_discriminator2_true = []
    #     self.lst_loss_epoch_discriminator2_fake = []
    #     self.lst_loss_epoch_generator = []
    #     return statistical_figure

    def plot_statistical_test(self, test_data, lst_gandalf_df_flux_scaled, lst_gandalf_wf_flux_scaled,
                              plot_name_statistical, plot_name_compare, batch_size, compare_plot=False, show=True,
                              epoch=10):
        """
        Plotting method to show the progress

        Args:
            plot_name: path and name of the plot image. If None, no plot image will be created. Default value is None
            show: If True, the plot image will be shown. Default value is True
        """
        self.lst_epochs.append(epoch)
        sns.set_theme()

        plt.close("all")

        statistical_figure, ((stat_ax1, stat_ax3, stat_ax4), (stat_ax2, stat_ax5, stat_ax6)) = plt.subplots(nrows=2, ncols=3)
        statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
        statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)

        # Create plot

        deep_field = reverses_standard_scaler(
            arr=test_data["deep field scaled"],
            mean=test_data["deep field mean"],
            std_dev=test_data["deep field standard deviation"])
        wide_field = reverses_standard_scaler(
            arr=test_data["wide field scaled"],
            mean=test_data["wide field mean"],
            std_dev=test_data["wide field standard deviation"])
        deep_field_generated = reverses_standard_scaler(
            arr=np.array(lst_gandalf_df_flux_scaled),
            mean=test_data["deep field mean"],
            std_dev=test_data["deep field standard deviation"])
        wide_field_generated = reverses_standard_scaler(
            arr=np.array(lst_gandalf_wf_flux_scaled),
            mean=test_data["wide field mean"],
            std_dev=test_data["wide field standard deviation"])

        self.lst_mean_diff.append(abs(np.mean(deep_field)-np.mean(deep_field_generated))/np.mean(deep_field))
        self.lst_stddev_diff.append(abs(standard_deviation(deep_field, np.mean(deep_field))-standard_deviation(deep_field_generated, np.mean(deep_field_generated)))/standard_deviation(deep_field, np.mean(deep_field)))

        stat_ax1.plot(
            self.lst_epochs,
            self.lst_mean_diff,
            alpha=0.5,
            marker=".")
        stat_ax1.set_xlabel("Epoch", fontsize=10, loc='right')
        stat_ax1.set_ylabel("Difference in mean", fontsize=10, loc='top')
        stat_ax1.set_title("Mean")

        stat_ax2.plot(
            self.lst_epochs,
            self.lst_stddev_diff,
            alpha=0.5,
            marker=".")
        stat_ax2.set_xlabel("Epoch", fontsize=10, loc='right')
        stat_ax2.set_ylabel("Difference in standard deviation", fontsize=10, loc='top')
        stat_ax2.set_title("Standard Deviation")

        stat_ax3.plot(
            deep_field,
            wide_field,
            ".",
            color="#DD8452",
            alpha=0.5,
            label="wide field flux")
        stat_ax3.plot(
            deep_field_generated,
            wide_field_generated,
            ".",
            color="#55A868",
            alpha=0.5,
            label="generated wide field flux")
        stat_ax3.legend()
        # stat_ax3.set_ylim((18000, 48000))
        # stat_ax3.set_xlim((18000, 48000))
        stat_ax3.set_xlabel("Deep field flux", fontsize=10, loc='right')
        stat_ax3.set_ylabel("Wide field flux", fontsize=10, loc='top')
        stat_ax3.set_title(f"Test data result")

        df_distribution_wf_flux = pd.DataFrame({
            "generated wide field flux": wide_field_generated,
            "wide field flux": wide_field
        })
        sns.histplot(df_distribution_wf_flux, stat="probability", kde=True, kde_kws={"bw_adjust": 3},
                     ax=stat_ax4, palette=["#55A868", "#DD8452"])
        stat_ax4.set_xlabel("Wide field flux", fontsize=10, loc='right')
        stat_ax4.set_title(f"Probability of wide field fluxes")

        cdf_wide_field_generated = np.array(ECDF(wide_field_generated).x)
        cdf_wide_field_generated = cdf_wide_field_generated[np.isfinite(cdf_wide_field_generated)]
        cdf_wide_field = np.array(ECDF(wide_field).x)
        cdf_wide_field = cdf_wide_field[np.isfinite(cdf_wide_field)]

        df_cdf = pd.DataFrame({
            "difference of cdf": np.abs(cdf_wide_field_generated - cdf_wide_field) + np.mean(cdf_wide_field),
            "cdf wide field": cdf_wide_field,
            "cdf wide field generated": cdf_wide_field_generated,
        })
        df_cdf.plot(
            grid=True,
            ax=stat_ax5)
        stat_ax5.set_xlabel("ID", fontsize=10, loc='right')
        stat_ax5.set_ylabel("Wide field flux", fontsize=10, loc='top')
        stat_ax5.set_title(f"Cumulative distribution functions")

        kstest_stat = kstest(
            rvs=wide_field,
            cdf=wide_field_generated
        )
        self.lst_critical_values.append(calc_ks_critical_value(len(wide_field), len(wide_field_generated), a=0.1))
        self.lst_kstest.append(kstest_stat[0])

        stat_ax6.plot(
            self.lst_epochs,
            self.lst_kstest,
            alpha=0.5,
            marker=".",
            label=f"Kolmogorov–Smirnov value: {self.lst_kstest[-1]:5.4f}")
        stat_ax6.plot(
            self.lst_epochs,
            self.lst_critical_values,
            alpha=0.5,
            marker=".",
            label=f"Kolmogorov–Smirnov critical value: {self.lst_critical_values[-1]:5.4f}")
        stat_ax6.set_xlabel("Epoch", fontsize=10, loc='right')
        stat_ax6.set_ylabel("Kolmogorov–Smirnov value", fontsize=10, loc='top')
        stat_ax6.set_title("Kolmogorov–Smirnov test")
        stat_ax6.legend()

        # If shown==True, plot image will be shown. Else pass
        # show = True
        if show is True:
            plt.show()
            exit()

        # If plot image is not None. The plot image will be saved with given path. Else pass
        if plot_name_statistical is not None:
            statistical_figure.savefig(plot_name_statistical, dpi=200)

        if compare_plot is True and plot_name_compare is not None:
            statistical_figure.suptitle(f"Epochs: {epoch}; "
                                        f"Input Tensor {self.input_dim}; "
                                        f"Learning rate discriminator {self.learning_rate_discriminator}; "
                                        f"Learning rate generator {self.learning_rate_generator}; "
                                        f"Batch size {batch_size}", fontsize=16)
            statistical_figure.savefig(plot_name_compare, dpi=200)

        # Clear and close open figure to avoid memory overload
        statistical_figure.clf()
        plt.close(statistical_figure)

        lst_data_types = []
        lst_x_values = []
        lst_y_values = []
        for idx, item in enumerate(deep_field_generated):
            lst_data_types.append("generated data")
            lst_x_values.append(item)
            lst_y_values.append(wide_field_generated[idx])

            lst_data_types.append("real data")
            lst_y_values.append(wide_field[idx])
            lst_x_values.append(deep_field[idx])

        self.lst_loss_epoch_discriminator_true = []
        self.lst_loss_epoch_discriminator_fake = []
        self.lst_loss_epoch_discriminator2_true = []
        self.lst_loss_epoch_discriminator2_fake = []
        self.lst_loss_epoch_generator = []
        return statistical_figure

    def save_network(self, filename_discriminator, filename_generator):
        torch.save(self.D, filename_discriminator)
        torch.save(self.G, filename_generator)

    def test_network(self, filename, filespath, outputpath, epochs, run, n, batch_size, plotting_statistical_test=False):
        """
        Function to test the gan with Test values and to verify how high the standard deviation is.

        Args:
            filename: Path of the *.pkl-file with the training data
            lr: learning rate in float
            G: Trained generator generator
            plotting_result: True to plot the result
            plotting_standard_deviation: True to plot the standard deviation

        Returns:
        """
        compare_plot = False
        plot_name_compare = None

        # Set output paths
        path_gan_test_output = f"{outputpath}/GAN_test_input_tensor_{run}_batch_size_{batch_size}/" \
                               f"lr_generator_{self.learning_rate_generator}_" \
                               f"lr_discriminator_{self.learning_rate_discriminator}"
        path_gan_test_plots = f"{path_gan_test_output}/plots"
        path_gan_test_loss = f"{path_gan_test_output}/loss"
        path_gan_test_mean_loss = f"{path_gan_test_output}/mean_loss"
        path_gan_test_result = f"{path_gan_test_output}/result"
        path_gan_save_path = f"{path_gan_test_output}/save_nn"
        path_gan_test_target = f"{path_gan_test_output}/target"
        path_gan_compare = f"{outputpath}/compare"

        if plotting_statistical_test is True:
            # Create folder if they doesn't exist
            if not os.path.exists(f"{outputpath}/GAN_test_input_tensor_{run}_batch_size_{batch_size}"):
                os.mkdir(f"{outputpath}/GAN_test_input_tensor_{run}_batch_size_{batch_size}")
            if not os.path.exists(path_gan_compare):
                os.mkdir(path_gan_compare)
            if not os.path.exists(path_gan_test_output):
                os.mkdir(path_gan_test_output)
            if not os.path.exists(path_gan_test_plots):
                os.mkdir(path_gan_test_plots)
            if not os.path.exists(path_gan_test_loss):
                os.mkdir(path_gan_test_loss)
            if not os.path.exists(path_gan_test_mean_loss):
                os.mkdir(path_gan_test_mean_loss)
            if not os.path.exists(path_gan_test_result):
                os.mkdir(path_gan_test_result)
            if not os.path.exists(path_gan_save_path):
                os.mkdir(path_gan_save_path)
            if not os.path.exists(path_gan_test_target):
                os.mkdir(path_gan_test_target)

        # Get the actual system time to calculate the duration from beginning
        start_time = datetime.now()

        train_data, test_data = load_simulated_data(filename)
        deviation_discriminator_true=0
        deviation_discriminator_fake=0
        deviation_discriminator2_true = 0
        deviation_discriminator2_fake = 0
        deviation_generator=0

        train_data2 = [0,1,2,3,4,5,6,7,8,9]
        test_data2 = [0,1,2,3,4,5,6,7,8,9]

        for i in np.arange(0,9):
            train_data2[i], test_data2[i]=load_simulated_data(filespath+f"\\{n}_{i+1}.pkl")

        # Start training
        for epoch in range(epochs):
            # Get the actual system time to calculate the duration for each epoch
            epoch_time = datetime.now()
            if epoch%1 == 0:
                deviation_discriminator_true, deviation_discriminator_fake, deviation_generator = self.train_on_batch(
                    inputs_discriminator=[train_data["wide field scaled"]],
                    labels_discriminator=[train_data["deep field scaled"]],
                    inputs_generator=[np.random.rand(len(train_data["deep field"]), self.input_dim)],
                    labels_generator=[train_data["deep field scaled"]],
                    batch_size=batch_size,
                    target_path=path_gan_test_target,
                    epoch=epoch
                )

                test_inputs_fake = [np.random.rand(len(test_data["deep field"]), self.input_dim)]
                test_labels_fake = [test_data["deep field scaled"]]

                if type(test_inputs_fake) is list:
                    test_inputs_fake = concatenate_lists(test_inputs_fake)
                if type(test_labels_fake) is list:
                    test_labels_fake = concatenate_lists(test_labels_fake)
                #stacked_test_input_fake = np.column_stack((test_inputs_fake, test_labels_fake))

                lst_gandalf_wf_flux_scaled = []
                lst_gandalf_df_flux_scaled = []
                for idx, test_item in enumerate(test_inputs_fake):
                    # Test the output of the gan with False values
                    gandalf_data = self.G.forward(torch.FloatTensor(test_item)).detach()
                    lst_gandalf_wf_flux_scaled.append(gandalf_data[0].item())
                    lst_gandalf_df_flux_scaled.append(gandalf_data[1].item())

            if epoch % 1 == 0:
                for i in np.arange(0,9):
                    deviation_discriminator2_true, deviation_discriminator2_fake, deviation_generator = self.train_on_batch2(
                        inputs_discriminator=[train_data2[i]["wide field scaled"]],
                        labels_discriminator=[train_data2[i]["deep field scaled"]],
                        inputs_generator=[np.random.rand(len(train_data2[i]["deep field"]), self.input_dim)],
                        labels_generator=[train_data2[i]["deep field scaled"]],
                        batch_size=batch_size,
                        target_path=path_gan_test_target,
                        epoch=epoch
                    )

                test_inputs_fake = [np.random.rand(len(test_data["deep field"]), self.input_dim)]
                test_labels_fake = [test_data["deep field scaled"]]

                if type(test_inputs_fake) is list:
                    test_inputs_fake = concatenate_lists(test_inputs_fake)
                if type(test_labels_fake) is list:
                    test_labels_fake = concatenate_lists(test_labels_fake)
                stacked_test_input_fake = np.column_stack((test_inputs_fake, test_labels_fake))

                lst_gandalf_wf_flux_scaled = []
                lst_gandalf_df_flux_scaled = []
                for idx, test_item in enumerate(test_inputs_fake):
                    # Test the output of the gan with False values
                    gandalf_data = self.G.forward(torch.FloatTensor(test_item)).detach()
                    lst_gandalf_wf_flux_scaled.append(gandalf_data[0].item())
                    lst_gandalf_df_flux_scaled.append(gandalf_data[1].item())

            # Plotting the progress if True
            if plotting_statistical_test is True:
                # Plotting the progress
                if epoch + 1 == epochs:
                    compare_plot = True
                    plot_name_compare = f"{path_gan_compare}/it_{self.input_dim}_bs_{batch_size}"\
                                        f"_lrg_{self.learning_rate_generator}"\
                                        f"_lrd_{self.learning_rate_discriminator}.png"
                self.plot_statistical_test(
                    test_data=test_data,
                    lst_gandalf_df_flux_scaled=lst_gandalf_df_flux_scaled,
                    lst_gandalf_wf_flux_scaled=lst_gandalf_wf_flux_scaled,
                    plot_name_statistical=f"{path_gan_test_plots}/epoch_{epoch+1}.png",
                    plot_name_compare=plot_name_compare,
                    batch_size=batch_size,
                    show=False,
                    epoch=epoch+1,
                    compare_plot=compare_plot)

            # Calculate the time difference
            delta_start_time = datetime.now() - start_time
            delta_epoch_time = datetime.now() - epoch_time
            print(f"Run = {run}\t"
                  f"Learning Rate Discriminator = {self.learning_rate_discriminator}\t"
                  f"Learning Rate Discriminator-2 = {self.learning_rate_discriminator * 1}\t"
                  f"Learning Rate Generator = {self.learning_rate_generator}\t")
            print(f"Elapsed time since start = {delta_start_time.seconds}s")
            print(f"Elapsed time for epoch {epoch} = {delta_epoch_time.seconds}s", )
            print(f"Deviation from equilibrium for discriminator true = {deviation_discriminator_true}")
            print(f"Deviation from equilibrium for discriminator fake = {deviation_discriminator_fake}")
            print(f"Deviation from equilibrium for discriminator-2 true = {deviation_discriminator2_true}")
            print(f"Deviation from equilibrium for discriminator-2 fake = {deviation_discriminator2_fake}")
            print(f"Deviation from equilibrium for generator = {deviation_generator}")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # self.save_network(
        #         f"{path_gan_save_path}/discriminator_epoch_{epoch+1}",
        #         f"{path_gan_save_path}/generator_epoch_{epoch+1}"
        #     )


if __name__ == "__main__":
    input_dim = 100
    lr = 0.0001
    GAN = GenerativeAdversarialNetwork(lr=lr, input_dim=input_dim)
    D = Discriminator(lr=lr)
    batch_size = 1
    epochs = 100
    path = os.path.abspath(sys.path[0])
    filepath = path + r"\..\Data"
    filespath = path+ r"\..\Data\lgsd"
    filename = filepath + r"\norm_simulated_data_different_noise_5000.pkl"
    outputpath = path + r"\..\Output"
    train_data, test_data = load_simulated_data(filename)

    GAN.test_network(
        filename=filename,
        outputpath=outputpath,
        epochs=epochs,
        batch_size=batch_size,
        plotting_statistical_test=True
    )
