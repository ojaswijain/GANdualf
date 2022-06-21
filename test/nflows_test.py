import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import kstest

np.random.seed(0)

lst_kstest = []
lst_kspvalue = []
max_i = 1000
# x = 0
# y = 0

for i in range(max_i):
    x = np.random.randn(1000)
    y = np.random.rand(100)

    kstest_stat = kstest(rvs=x, cdf=y)
    lst_kstest.append(kstest_stat[0])
    lst_kspvalue.append(kstest_stat[1])


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


ks_figure, ((ks_ax1, ks_ax2)) = plt.subplots(nrows=2)
ks_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
ks_figure.suptitle(f"KS example", fontsize=16)

mean_kstest = np.mean(lst_kstest)
critical_value = calc_ks_critical_value(len(x), len(y))
ks_ax1.plot(lst_kstest, ".", label=f"KS statistics")
ks_ax1.plot([0, max_i], [critical_value, critical_value], label=f"Critical value = {critical_value}")
ks_ax1.plot([0, max_i], [mean_kstest, mean_kstest], label=f"Mean KS Test = {mean_kstest}")
ks_ax1.legend()
ks_ax1.set_xlabel("no of test", fontsize=10, loc='right')
ks_ax1.set_ylabel("ks statistic", fontsize=10, loc='top')
ks_ax1.set_title(f"KS statistic test")

mean_kspvalue = np.mean(lst_kspvalue)
ks_ax2.plot(lst_kspvalue, ".", label=f"KS p-Value")
ks_ax2.plot([0, max_i], [mean_kspvalue, mean_kspvalue], label=f"Mean KS p-value = {mean_kspvalue}")
ks_ax2.legend()
ks_ax2.set_xlabel("no of test", fontsize=10, loc='right')
ks_ax2.set_ylabel("ks p-value", fontsize=10, loc='top')
ks_ax2.set_title(f"KS p-value")

plt.show()
