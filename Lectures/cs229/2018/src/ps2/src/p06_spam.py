import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split()
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counter = {}
    for message in messages:
        message_ = get_words(message)

        for word in message_:
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1

    my_dict = {}
    idx = 0
    for k in counter.keys():
        if counter[k] >= 5:
            my_dict[k] = idx
            idx += 1

    return my_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    l_m = len(messages)
    l_d = len(word_dictionary)

    result = np.zeros((l_m, l_d+1)) # (4457, 1758)

    for i, message in enumerate(messages):
        message_ = get_words(message)
        for word in message_:
            if word in word_dictionary:
                result[i][word_dictionary[word]] += 1


    return result
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    m,n = matrix.shape

    pi_1 = np.zeros(n) # 1758
    pi_0 = np.zeros(n) # 1758

    rows_1 = matrix[labels == 1] # (611, 1758)
    rows_0 = matrix[labels == 0] # (3846, 1758)

    deno_1 = rows_1.shape[0] * rows_1.shape[1]
    deno_0 = rows_0.shape[0] * rows_0.shape[1]

    for i, v in enumerate(pi_1):
        pi_1[i] = rows_1[:,i].sum()

    for i, v in enumerate(pi_0):
        pi_0[i] = rows_0[:,i].sum()

    # applying laplace smoothing: 1,v
    pi_1 = (pi_1+1)/(deno_1+n)
    lpi_1 = np.log10(pi_1)
    pi_0 = (pi_0+1)/(deno_0+n)
    lpi_0 =np.log10(pi_0)

    # print(lpi_1)
    # print(lpi_0)

    lp_1 = np.log10(rows_1.shape[0] / m)
    lp_0 = np.log10(rows_0.shape[0] / m) # 1 - lp_1

    return [lpi_1, lpi_0, lp_1, lp_0]
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    lpi_1, lpi_0, lp_1, lp_0 = model
    m,n = matrix.shape

    predict = np.zeros(m)

    for index, row in enumerate(matrix):
        l1, l0 = lp_1, lp_0
        for i, c in enumerate(row):
            if c > 0:
                l1 += lpi_1[i]
                l0 += lpi_0[i]

        if l1 > l0:
            predict[index] = 1
        else:
            predict[index] = 0

    # print(predict)
    # print(predict.shape)

    return predict
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    lpi_1, lpi_0, lp_1, lp_0 = model # 1758
    dtype = [('index', int), ('indicator', float)]
    result = []

    for idx in range(len(lpi_1)):
        indicator = lpi_1[idx] - lpi_0[idx]
        result.append((idx, indicator))

    a = np.array(result, dtype=dtype)  # create a structured array
    a = np.sort(a, order='indicator')


    words = []
    new_dict = dict([(value, key) for key, value in dictionary.items()])

    for i, v in a[-5:]:
        words.append(new_dict[i])

    return words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***

    highest_accuracy = -1
    highest_accuracy_radius = -1

    for radius in radius_to_consider:
        svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        svm_accuracy = np.mean(svm_predictions == val_labels)
        if svm_accuracy > highest_accuracy:
            highest_accuracy = svm_accuracy
            highest_accuracy_radius = radius

    return highest_accuracy_radius
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
