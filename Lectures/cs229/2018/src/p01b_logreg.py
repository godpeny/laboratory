import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    y_hat = clf.predict(x_eval)

    cnt = 0
    for y_hat_element, y_element in zip(y_hat, y_eval):
        pred = 0
        if y_hat_element > 0.5:
            pred = 1
        if pred != y_element:
            cnt += 1

    print("error rate : ", cnt/len(y_hat))

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])  # (1,3)
        self.eps = 0.00001

        while True:
            g_theta_x = 1 / (1 + np.exp(np.dot(-x, self.theta.T)))  # (800, 1) = (800, 3) @ (3, 1)

            hessian = x @ x.T @ g_theta_x * (1 - g_theta_x)  # (800, 1) = (800, 3) @ (3, 800) @ (800, 1) * (800, 1)
            hessian = np.average(hessian)  # scalar

            gradient = -x.T @ (y - g_theta_x)  # (3, 1) = (3, 800) @ (800, 1)

            theta_updated = self.theta - (gradient/hessian)

            gap = np.average(np.abs(theta_updated - self.theta))
            print("theta : ", theta_updated, " gap : ", gap)
            if gap < self.eps:
                break

            self.theta = theta_updated
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta.T)))  # logistic regression
        # *** END CODE HERE ***
