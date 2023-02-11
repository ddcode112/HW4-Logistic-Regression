import sys
import csv
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

if __name__ == '__main__':
    trainFile = sys.argv[1]
    validationFile = sys.argv[2]
    testFile = sys.argv[3]
    featureDict = sys.argv[4]
    trainOutFile = sys.argv[5]
    validationOutFile = sys.argv[6]
    testOutFile = sys.argv[7]


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def exclude_word(word_matrix, keys):
    trim_word_matrix = []
    for row in word_matrix:
        new_string = ""
        word = row[1].split(" ")

        for w in word:
            if w in keys:
                new_string = new_string + w + " "
        new_string = new_string.strip()
        trim_word_matrix.append((row[0], new_string))
        trim_word_matrix_arr = np.array(trim_word_matrix)
    return trim_word_matrix_arr


def convert_feature_vector(word_matrix, feature_dictionary):
    feature_vector = []
    for i in word_matrix:
        w = np.array(i[1].split(" "))
        word, counts = np.unique(w, return_counts=True)
        word_feature = np.array([feature_dictionary.get(v) for v in word])
        i_feature_vector = np.matmul(counts, word_feature)
        i_feature_vector = i_feature_vector/len(w)
        feature_vector.append(i_feature_vector)
    feature_vector_arr = np.array(feature_vector)
    return feature_vector_arr


def formatted_output(labels, feature_vector, filepath):
    labels = labels.astype(float)
    output_matrix = np.insert(feature_vector, 0, labels, axis=1)
    np.savetxt(filepath, output_matrix, delimiter='\t', fmt='%.6f')


train_input = load_tsv_dataset(trainFile)
validation_input = load_tsv_dataset(validationFile)
test_input = load_tsv_dataset(testFile)
feature_dictionary_input = load_feature_dictionary(featureDict)

feature_dictionary_key = feature_dictionary_input.keys()
trim_train_input = exclude_word(train_input, feature_dictionary_key)
train_feature_vector = convert_feature_vector(trim_train_input, feature_dictionary_input)
trim_validation_input = exclude_word(validation_input, feature_dictionary_key)
validation_feature_vector = convert_feature_vector(trim_validation_input, feature_dictionary_input)
trim_test_input = exclude_word(test_input, feature_dictionary_key)
test_feature_vector = convert_feature_vector(trim_test_input, feature_dictionary_input)

formatted_output(trim_train_input[:, 0], train_feature_vector, trainOutFile)
formatted_output(trim_validation_input[:, 0], validation_feature_vector, validationOutFile)
formatted_output(trim_test_input[:, 0], test_feature_vector, testOutFile)



