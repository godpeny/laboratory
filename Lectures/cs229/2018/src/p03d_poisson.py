import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    poisson = PoissonRegression(step_size=lr)
    poisson.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    y_pred = poisson.predict(x_eval)

    # save
    np.savetxt(pred_path, y_pred)

    plt.plot(y_eval, 'go', label='label')
    plt.plot(y_pred, 'rx', label='prediction')
    plt.suptitle("xx", fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape  # (2500, 4)
        y = y.reshape(-1, 1)  # (2500, 1)
        self.theta = np.zeros([1, n])  # (1, 4)
        self.eps = 0.00001

        while True:
            e = np.exp(x @ self.theta.T)  # (2500, 1) = (2500, 4) @ (4, 1)
            derivative = ((y - e).T @ x) / m  # (1, 4)
            theta_new = self.theta + (self.step_size * derivative)  # gradient ascent

            gap = np.abs(np.mean(theta_new - self.theta))
            print("theta : ", theta_new, " gap : ", gap)
            if gap < self.eps:
                break
            self.theta = theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta.T)  # lambda = expected value
        # *** END CODE HERE ***
