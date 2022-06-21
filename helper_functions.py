import numpy as np
import torch
import pandas as pd


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
        return np.random.uniform(low, high, size=(num, size))
    return torch.FloatTensor(np.random.uniform(low, high, size=(num, size)))


def calc_ks_critical_value(n1, n2, a=0.10):
    if a == 0.10:
        c_a = 1.22
    elif a == 0.05:
        c_a = 1.36
    elif a == 0.025:
        c_a = 1.48
    elif a == 0.01:
        c_a = 1.63
    elif a == 0.005:
        c_a = 1.73
    elif a == 0.001:
        c_a = 1.95
    else:
        raise Exception("Wrong value for a. a must be one of these values [0.10, 0.05, 0.025, 0.01, 0.005, 0.001]")
    return c_a * np.sqrt((n1+n2)/(n1*n2))


def standard_deviation(arr, mean):
    """
    Calculate the standard deviation

    Args:
        arr: array of values

    Returns:
        the calculated standard deviation as array
    """
    return np.sqrt(np.sum((arr - mean)**2) / len(arr))


def standard_scaler(arr, mean, std_dev):
    """
    Calculate the standard scale

    Args:
        arr: array of values

    Returns:
        calculated standard scale as array
    """
    return (arr - mean) / std_dev


def reverses_standard_scaler(arr, mean, std_dev):
    """
    reverses the standard scale

    Args:
        arr: array of values

    Returns:
        calculated original array
    """
    return std_dev * arr + mean


def normalizing(values):
    return values / values.max()


def concatenate_lists(data_list):
    conc_inputs = data_list[0]
    for idx, value in enumerate(data_list):
        if idx + 1 < len(data_list):
            conc_inputs = np.concatenate((conc_inputs, data_list[idx + 1]), axis=0)
    return conc_inputs
