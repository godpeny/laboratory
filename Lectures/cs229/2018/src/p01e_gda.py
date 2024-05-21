import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)
    clf.predict(x_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # phi
        sum_y_1 = 0
        for y_i in y:
            if y_i == 1:
                sum_y_1 += 1
        phi = sum_y_1 / len(y)

        # m_0
        sum_y_0 = 0
        for y_i in y:
            if y_i == 0:
                sum_y_0 += 1

        sum_x_y_0 = 0
        for x_i, y_i in zip(x, y):
            if y_i == 0:
                sum_x_y_0 += x_i

        m_0 = sum_x_y_0 / sum_y_0

        # m_1
        sum_x_y_1 = 0
        for x_i, y_i in zip(x, y):
            if y_i == 1:
                sum_x_y_1 += x_i

        m_1 = sum_x_y_1 / sum_y_1

        # sig
        sig_tot = 0
        for x_i, y_i in zip(x, y):
            if y_i == 0:
                m = m_0
            else:
                m = m_1

            sig_tot += (x_i - m).dot((x_i - m).T)

        sig = sig_tot / len(y)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
