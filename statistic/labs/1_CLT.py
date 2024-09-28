import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as dist


def main():
    # будем моделировать количество орлов в серии из 30 подбрасываний
    generate_sample(10)
    generate_sample(50)
    generate_sample(100)
    generate_sample(1000)
    generate_sample(10000)


def generate_sample(n):
    plt.figure(figsize=(8, 4), dpi=100)
    fig, ax1 = plt.subplots()

    p = 0.5
    num = 30
    rv_binom = dist.binom(p=p, n=num)
    sample_n = rv_binom.rvs(n)

    ax1.hist(sample_n, bins=30)

    ax2 = ax1.twinx()
    x = np.linspace(sample_n.min(), sample_n.max(), 100)
    ax2.plot(x, dist.norm.pdf(x, rv_binom.mean(), rv_binom.std()), 'r')

    fig.savefig(f"images/lab1/sample_{n}.png", dpi=200)
    plt.clf()


if __name__ == '__main__':
    main()
