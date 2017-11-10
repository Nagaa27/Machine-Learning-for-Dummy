# Matius Celcius Sinaga
# Ubuntu 14.0 | Python27

import numpy as np
import matplotlib.pyplot as plt
import multivariasi


def main():
    X = multivariasi.load_norm_data()

    plt.scatter(X[:1000, 0], X[:1000, 1], color='r', marker='x',
                label='$dist_1$')
    plt.scatter(X[1000:2000, 0], X[1000:2000, 1], color='b', marker='x',
                label='$dist_2$')
    plt.show()

if __name__ == '__main__':
    main()
    

