import information_coefficient

from scipy.stats import pearsonr
import numpy as np
from math import exp, pi, log, sqrt
from random import randint

EPS = np.finfo(float).eps

RANDOM_SEED = 20121020
default = {
    "n_grids"       : 25,
    "jitter"        : 1E-10,
    "random_seed"   : RANDOM_SEED,
    "bandwidth_mult": 1.0,
    "k"             : 1
}
def binaryInformationCoefficient(y, x, options=default):
    """ Calculate IC between continuous 'y' and binary 'x' """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x)

    # Calculate bandwidth
    cor, _ = pearsonr(y, x)
    y_bandwidth = information_coefficient.calc_bandwidth(y)
    y_bandwidth *= (options["bandwidth_mult"] * (1 + (-0.75) * abs(cor)))

    # Prepare grids
    y_total = np.zeros(options["n_grids"])
    y_0 = np.zeros(options["n_grids"])
    y_1 = np.zeros(options["n_grids"])
    k = options["k"]
    dy = (y.max() - y.min())/options["n_grids"]
    sigma_y = y_bandwidth/dy
    term_2_pi_sigma_y = 2 * pi * sigma_y
    kernel = { i: exp(-0.5*((i/sigma_y)**2))/term_2_pi_sigma_y for i in range(-k, k+1) }

    # Calculate probability density
    for i, val in enumerate(y):
        y_d = int(round((val - y.min())/dy)) + k
        for j in range(-k, k+1):
            index = y_d + j
            if index >= options["n_grids"]:
                continue
            y_total[index] = kernel[j]
            if 0 == x[i]:
                y_0[index] = kernel[j]
            else:
                y_1[index] = kernel[j]

    # Normalize
    for array in (y_total, y_0, y_1):
        array += EPS
        array = array / (array.sum())
    
    # Calculate mutual information
    mutual_information = 0
    dy = (y.max() - y.min()) / (options["n_grids"] - 1)
    for p_x in (y_0, y_1):
        integral = sum([p_x[i] * log(p_x[i]/y_total[i]) for i in range(options["n_grids"])]) * dy
        print("integral", integral)
        mutual_information += sum(p_x) * integral

    # TODO: debug when MI < 0 and |MI| ~0 resulting in IC = nan
    if mutual_information < 0 and abs(mutual_information) < 0.1:
        return 0

    return np.sign(cor) * sqrt(1 - exp(-2 * mutual_information))

def test():
    y = [randint(0, 100) for i in range(1000)]
    x = [1 if y_i >= 50 else 0 for y_i in y]
    # x = [randint(0, 1) for y_i in y]
    print("y:", y)
    print("x:", x)
    print(binaryInformationCoefficient(y, x))

if "__main__" == __name__:
    test()