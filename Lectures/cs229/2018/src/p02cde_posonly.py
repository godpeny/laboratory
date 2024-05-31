import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    model = LogisticRegression()

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    t_hat = model.fit(x_train, t_train)
    util.plot(x_test, t_test, model.theta.squeeze(axis=0), '{}.png'.format(pred_path_c[:-4]))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    _, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    y_hat = model.fit(x_train, y_train)
    util.plot(x_test, y_test, model.theta.squeeze(axis=0), '{}.png'.format(pred_path_d[:-4]))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
