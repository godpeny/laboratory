# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape
    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y)) + (0.000001*theta)
    # L2 Norm
    # add derivative of L2 Norm of theta (= theta**2) respect to theta_k to the derivate of cost function
    # lambda = 0.000001
    # grad = -(1. / m) * (X.T.dot(probs * Y)) + (0.000001 * theta)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    # zero mean normal distribution noise
    # np.random.normal(0, 0.1, X.shape)
    # X += noise

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(np.linalg.norm(prev_theta - theta))
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def data_plot():
    x, y = util.load_csv('../data/ds1_a.csv')

    # Create a scatter plot
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x[i][0], x[i][1], marker='o', color='blue')
        elif y[i] == -1:
            plt.scatter(x[i][0], x[i][1], marker='x', color='red')

    # Add labels
    plt.xlabel('x_0')
    plt.ylabel('x_1')

    # Show plot
    plt.title('Scatter plot of x_0 vs x_1 with y labels')
    plt.grid(True)
    plt.show()

    x, y = util.load_csv('../data/ds1_b.csv')

    # Create a scatter plot
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x[i][0], x[i][1], marker='o', color='blue')
        elif y[i] == -1:
            plt.scatter(x[i][0], x[i][1], marker='x', color='red')

    # Add labels
    plt.xlabel('x_0')
    plt.ylabel('x_1')

    # Show plot
    plt.title('Scatter plot of x_0 vs x_1 with y labels')
    plt.grid(True)
    plt.show()

def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
    # data_plot()
