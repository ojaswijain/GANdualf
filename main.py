from gan import GenerativeAdversarialNetwork
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(__file__))

plt.rcParams["figure.figsize"] = (16, 9)


def main(root_path, filename, filespath, outputpath, learning_rate_generator, learning_rate_discriminator, input_dim_generator,
         epochs, run, batch_size, plotting_statistical_test,n):
    if not os.path.exists(f"{root_path}\save_nn"):
        os.mkdir(f"{root_path}\save_nn")
    if not os.path.exists(f"{root_path}\Output"):
        os.mkdir(f"{root_path}\Output")

    GAN = GenerativeAdversarialNetwork(lr_g=learning_rate_generator, lr_d=learning_rate_discriminator,
                                       input_dim=input_dim_generator)

    GAN.test_network(
        filename=filename,
        filespath=filespath,
        outputpath=outputpath,
        epochs=epochs,
        run=run,
        batch_size=batch_size,
        n=n,
        plotting_statistical_test=plotting_statistical_test)


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])

    lst_learning_rate_generator = [1E-7] # 5E-6, , 5E-7
    lst_learning_rate_discriminator = [1E-5]  # 9E-4, 5E-4, , 5E-5, 9E-5
    lst_input_tensors = [10]  # 5, 10, 20, 50, 100, 150, 200, , 1000
    lst_batch_sizes = [128]  # 2, 4, 8, 16, 32, , 128, 256

    for i in lst_learning_rate_generator:
        for j in lst_learning_rate_discriminator:
            for k in lst_input_tensors:
                for n in lst_batch_sizes:
                    main(root_path=path,
                         filename=path + r"\Data\linear_gauss_2000.pkl",
                         outputpath=path + r"\Output",
                         filespath=path + r"\Data\lgsd",
                         learning_rate_generator=i,
                         learning_rate_discriminator=j,
                         input_dim_generator=k,
                         epochs=100,
                         run=k,
                         batch_size=n,
                         n=2000,
                         plotting_statistical_test=True)
