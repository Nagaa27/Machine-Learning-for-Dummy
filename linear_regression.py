# Matius Celcius Sinaga
# Ubuntu 14.0 | Python27

import multivariasi
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(theta, X, y):
    m = len(X)
    J = np.sum(
        [(np.dot(theta.T, X[i]) - y[i]) ** 2 for i in range(m)]
    )
    return (1./(2*m)) * J


def gradient_descent(theta, X, y, alpha, max_iter):
    m = len(X)
    num_params = len(theta)
    grad = np.zeros([num_params, 1])

    iter = 0
    while iter < max_iter:
        for j in range(num_params):
            grad[j] = np.sum(
                [(np.dot(theta.T, xx) - yy) * xx[j] for (xx, yy) in zip(X, y)]
            )
            grad[j] = (alpha/m) * grad[j]
            theta[j] -= grad[j]
        iter += 1

    return theta


def main():
    #load sample data
    #mengisi contoh data
    data = multivariasi.load_data_single()
    X_, y = data[:, 0], data[:, 1]
    X = np.ones([y.size, 2])
    X[:, 1] = X_

    #compute theta
    #mengitung theta
    m, dim = X.shape
    theta = np.zeros([dim, 1])
    alpha, max_iter = 0.01, 300
    theta = gradient_descent(theta, X, y, alpha, max_iter)
    print theta

    #plot sample data and predicted line
    #plot sample data dan garis yang diprediksi
    plt.subplot(2, 1, 1)
    plt.scatter(data[:, 0], data[:, 1], color='r', marker='x')
    xx = np.linspace(-10, 10)
    yy = theta[0] + theta[1] * xx
    plt.plot(xx, yy, 'k-')

    #plot contour
    #bentuk permukaan plot
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    #initialize J_vals to a matrix of 0's
    #mengenalkan J_vals pada matrix dari 0
    J_vals = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

    #fill out J_vals
    #mengisi keluaran J_vals
    for t1, element in enumerate(theta0_vals):
        for t2, element2 in enumerate(theta1_vals):
            thetaT = np.zeros(shape=(2, 1))
            thetaT[0][0] = element
            thetaT[1][0] = element2
            J_vals[t1, t2] = compute_cost(thetaT, X, y)

    #contour plot
    #bentuk permukaan plot
    J_vals = J_vals.T
    #plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    #plot J_vals memiliki 15 bentuk permukaan yang trisi secara logaritma 
    #diantara 0.01 dan 100
    plt.subplot(2, 1, 2)
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 40))
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.scatter(theta[0][0], theta[1][0])
    plt.show()


if __name__ == '__main__':
    main()
