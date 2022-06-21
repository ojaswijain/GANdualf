import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_catalog(n_sources, transfer_type, seed):
    sources = np.random.normal(30000, 5000, size=n_sources)# [np.random.randn(20000, 40000) for _ in range(0, n_sources)]
    df_sources = None
    if transfer_type == "linear":
        df_sources = pd.DataFrame({"deep field": sources, "flux": linear_function(sources)})
    elif transfer_type == "linear gauss":
        df_sources = pd.DataFrame({"deep field": sources, "flux": linear_function_with_gauss(sources, seed=seed)})
    elif transfer_type == "function":
        df_sources = pd.DataFrame({"deep field": sources, "flux": transfer_function(sources)})
    elif transfer_type == "function gauss":
        df_sources = pd.DataFrame({"deep field": sources, "flux": transfer_function_with_gauss(sources, seed=seed)})
    elif transfer_type == "two gaussian distribution":
        df_sources = pd.DataFrame({"deep field": np.random.randn(n_sources), "flux": np.random.randn(n_sources)})
    return sources, df_sources


def transfer_function(arr):
    return arr*(arr-26541)/1e4+np.random.randn()*4000


def transfer_function_with_gauss(arr, seed):
    if seed is True:
        np.random.seed(123)
    try:
        lst = []
        for value in arr:
            if value < 25000:
                lst.append((-value * (value - 26541) * (value - 35416) * (value - 38215))/1e12+np.random.randn()*4000)
            elif 25000 <= value <= 35000:
                lst.append((-value * (value - 26541) * (value - 35416) * (value - 38215)) / 1e12 + np.random.randn() * np.random.poisson()*3000)
            elif 35000 < value:
                lst.append(
                    (-value * (value - 26541) * (value - 35416) * (value - 38215)) / 1e12 + np.random.randn()*3000 + np.random.rand()*500)
        return lst
    except TypeError:
        return (-arr * (arr - 26541) * (arr - 35416) * (arr - 38215))/1e4+np.random.randn()*4000


def linear_function(arr):
    return arr*1.12


def linear_function_with_gauss(arr, seed):
    if seed is True:
        np.random.seed(123)
    try:
        lst = []
        for value in arr:
            lst.append(value*1.22**2 + 500*(np.random.randn() + np.random.randn()*1.1 + np.random.poisson()*0.89 * np.random.randn()**2))
        return lst
    except TypeError:
        return arr * 1.12 + np.random.randn() * 1000


if __name__ == "__main__":
    number_of_sources = [200, 2000, 5000, 10000, 20000]
    transfer_type = "linear gauss"
    seed = True
    show_plot = True

    for x in number_of_sources:

        src_cat, df_src_cat = make_catalog(
            n_sources=x,
            transfer_type=transfer_type,
            seed=seed)

        if show_plot is True:
            plt.plot(df_src_cat["deep field"], df_src_cat["flux"], ".")
            plt.show()
        # df_src_cat.to_csv(f"simulated_data_linear_gaussian_{number_of_sources}.csv")
        pkl_sources = df_src_cat.to_pickle(f"linear_gauss_{x}.pkl")
