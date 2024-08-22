import math

import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    tau_values = [3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1]
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    lowest_mse = math.inf
    best_y_pred = np.zeros(x_eval.shape[0],)

    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau=tau)
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_eval)

        mse = np.mean((y_eval - y_pred)**2)

        if mse < lowest_mse:
            lowest_mse = mse
            best_y_pred = y_pred

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, best_y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')

    print(lowest_mse)
    plt.show()
    # *** END CODE HERE ***
