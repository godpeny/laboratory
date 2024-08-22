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
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model_t = LogisticRegression()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_t.fit(x_train, t_train)
    t_hat = model_t.predict(x_train)

    util.plot(x_test, t_test, model_t.theta.squeeze(axis=0), '{}.png'.format(pred_path_c[:-4]))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model_y = LogisticRegression()
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    model_y.fit(x_train, y_train)
    y_hat = model_y.predict(x_train)

    util.plot(x_test, t_test, model_y.theta.squeeze(axis=0), '{}.png'.format(pred_path_d[:-4]))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_val, t_val = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    _, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    y_val_true_idx = np.where(y_val == 1)[0]
    t_val_true = t_val[y_val_true_idx]
    x_val_true = x_val[y_val_true_idx]
    alpha = np.average(model_y.predict(x_val_true))
    
    # making new decistion boundary with theta, alpha and threshold given by problem.
    # since theta[0] is for intercept (bias), "np.log((2/alpha) - 1)" is added to theta[0].
    # since "theta_new = model_y.theta AND theta_new[0] + np.log((2/alpha) - 1)", corr should be as below.
    corr = (1 + np.log(2 / alpha - 1) / model_y.theta.squeeze(axis=0)[0])

    util.plot(x_test, t_test, model_y.theta.squeeze(axis=0), '{}.png'.format(pred_path_e[:-4]), corr)
    # *** END CODER HERE
