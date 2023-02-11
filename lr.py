import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    formatted_trainFile = sys.argv[1]
    formatted_validationFile = sys.argv[2]
    formatted_testFile = sys.argv[3]
    trainOutFile = sys.argv[4]
    testOutFile = sys.argv[5]
    metrics = sys.argv[6]
    num_epoch = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    learning_rate_4 = float(sys.argv[9])
    learning_rate_5 = float(sys.argv[10])


def load_tsv_dataset(file):

    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')

    return dataset


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(-x)
    return 1 / (1 + e)


def dJ(theta, X, y, i):
    s = sigmoid(np.matmul(theta.T, X[i]))
    j_i_gd = np.dot(s - y[i], X[i])
    return j_i_gd


def SGD(theta, X, val_X, y, val_y, num_epoch, learning_rate):
    train_log_likelihood = np.array([])
    val_log_likelihood = np.array([])
    for epoch in range(num_epoch):
        indices = len(X)
        for i in range(indices):
            theta -= learning_rate * dJ(theta, X, y, i)
        s = sigmoid(np.matmul(theta, X.T))
        s_val = sigmoid(np.matmul(theta, val_X.T))
        #print(-np.multiply(y, np.log(s)) - np.multiply((1 - y), np.log(1 - s)))
        train_log_likelihood = np.append(train_log_likelihood, np.mean(-np.multiply(y, np.log(s)) - np.multiply((1 - y), np.log(1 - s))))
        # print("train: ", np.mean(-np.multiply(y, np.log(s)) - np.multiply((1 - y), np.log(1 - s))))
        # print(-np.matmul(y, np.log(s)) - np.matmul((1 - y), np.log(1 - s)))
        val_log_likelihood = np.append(val_log_likelihood, np.mean(-np.multiply(val_y, np.log(s_val)) - np.multiply((1 - val_y), np.log(1 - s_val))))
        #print(-np.multiply(val_y, np.log(s_val)) - np.multiply((1 - val_y), np.log(1 - s_val)))
        # print("val: ", np.mean(-np.multiply(val_y, np.log(s_val)) - np.multiply((1 - val_y), np.log(1 - s_val))))
    return theta, train_log_likelihood, val_log_likelihood


def train(theta, X, val_X, y, val_y, num_epoch, learning_rate):
    theta, train_log_likelihood, val_log_likelihood = SGD(theta, X, val_X, y, val_y, num_epoch, learning_rate)
    return theta, train_log_likelihood, val_log_likelihood


def predict(theta, X):
    prediction = np.array([])
    for i in range(len(X)):
        if sigmoid(np.matmul(theta.T, X[i])) >= 0.5:
            prediction = np.append(prediction, 1)
        else:
            prediction = np.append(prediction, 0)
    return prediction


def compute_error(y_pred, y):

    error_list = [0 if y_pred[i] == y[i] else 1 for i in range(len(y_pred))]
    error = np.sum(error_list)

    return error/len(y)


def output_labels(labels, filepath):
    np.savetxt(filepath, labels, delimiter='\n', fmt='%i')


formatted_train_input = load_tsv_dataset(formatted_trainFile)
train_labels = np.array(formatted_train_input[:, 0])
formatted_train_input[:, 0] = np.ones(len(formatted_train_input))
print("train_labels shape: ", train_labels.shape)
print("train_input shape: ", formatted_train_input.shape)
formatted_validation_input = load_tsv_dataset(formatted_validationFile)
validation_labels = np.array(formatted_validation_input[:, 0])
formatted_validation_input[:, 0] = np.ones(len(formatted_validation_input))
print("validation_labels shape: ", validation_labels.shape)
print("validation_input shape: ", formatted_validation_input.shape)
formatted_test_input = load_tsv_dataset(formatted_testFile)
test_labels = np.array(formatted_test_input[:, 0])
formatted_test_input[:, 0] = np.ones(len(formatted_test_input))
print("test_labels shape: ", test_labels.shape)
print("test_input shape: ", formatted_test_input.shape)

theta = np.zeros(formatted_train_input.shape[1])
print("theta shape: ", theta.shape)

theta, train_log_likelihood, val_log_likelihood = train(theta, formatted_train_input, formatted_validation_input, train_labels, validation_labels, num_epoch, learning_rate)
train_prediction = predict(theta, formatted_train_input)
test_prediction = predict(theta, formatted_test_input)

train_error = compute_error(train_prediction, train_labels)
test_error = compute_error(test_prediction, test_labels)

output_labels(train_prediction, trainOutFile)
output_labels(test_prediction, testOutFile)

with open(metrics, 'w') as f_metrics:
    f_metrics.write(f'error(train): {train_error:.6f}' + '\n')
    f_metrics.write(f'error(test): {test_error:.6f}')

# theta_4, train_log_likelihood_4, val_log_likelihood_4 = train(np.zeros(formatted_train_input.shape[1]), formatted_train_input, formatted_validation_input, train_labels, validation_labels, num_epoch, learning_rate_4)
# theta_5, train_log_likelihood_5, val_log_likelihood_5 = train(np.zeros(formatted_train_input.shape[1]), formatted_train_input, formatted_validation_input, train_labels, validation_labels, num_epoch, learning_rate_5)

#print(train_log_likelihood.shape)
# print(val_log_likelihood)

x_arr = np.arange(1, 1001)
# plt.plot(x_arr, train_log_likelihood, label="\u03B7 = $10^{-3}$")
# plt.plot(x_arr, train_log_likelihood_4, label="\u03B7 = $10^{-4}$")
# plt.plot(x_arr, train_log_likelihood_5, label="\u03B7 = $10^{-5}$")
plt.plot(x_arr, train_log_likelihood, label="train log-likelihood")
plt.plot(x_arr, val_log_likelihood, label="validation log-likelihood")
plt.legend()
plt.ylabel('log-likelihood')
plt.xlabel('number of epochs')
plt.savefig('train_val_log_likelihood.png')


