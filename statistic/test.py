import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as dist


def main():
    mu = 175
    sigma = 20
    rv_norm = dist.norm(mu, sigma)
    x = np.arange(75, 275, 0.1)

    plt.plot(x, rv_norm.pdf(x))
    # plt.show()

    print(rv_norm.cdf(140))


if __name__ == '__main__':
    main()
