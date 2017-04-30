import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #print(train_data.shape)

    #Adding bias term to train_data matrix
    train_data = np.insert(train_data, 0, 1, axis=1)
    #print train_data.shape
    N = train_data.shape[0]
    initialWeights = np.reshape(initialWeights, (716,1))
    #print initialWeights.shape




    part_1 = np.dot(np.transpose(labeli), np.log(sigmoid(np.dot(train_data, initialWeights))))
    part_2 = np.dot(np.transpose(1 - labeli), np.log(np.subtract(1, sigmoid(np.dot(train_data, initialWeights)))))
    error = np.add(error, np.add(part_1, part_2))

    error = np.multiply(np.negative(np.reciprocal(float(N))), error)
    theta_matrix = sigmoid(np.dot(train_data, initialWeights))
    #print theta_matrix.shape
    sub_matrix = np.subtract(theta_matrix, labeli)
    #print sub_matrix.shape
    error_grad = np.dot(np.transpose(train_data), sub_matrix)

    error_grad = np.multiply(np.reciprocal(float(N)), error_grad)
    error_grad = np.ndarray.flatten(error_grad)


    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    # label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #print(train_data.shape)

    #Adding bias term to train_data matrix
    data = np.insert(data, 0, 1, axis=1)
    #print(train_data.shape)
    N = data.shape[0]

    theta_matrix = sigmoid(np.dot(data, W))
    label = np.argmax(theta_matrix, axis=1)
    print ("shape of label")
    print (label.shape)
    label = label.reshape(N,1)
    print (label)
    print ("SHAPE OF LABEL")
    print (label.shape)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Func for plotting only (using results from the file output_svm).
"""

def plot():
    _cvalues = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    _rbf_train_predicted = [94.293999999999997, 97.131999999999991, 97.951999999999998, 98.372, 98.706000000000003,
                            99.001999999999995, 99.195999999999998, 99.339999999999989, 99.438000000000002,
                            99.542000000000002, 99.611999999999995]
    _rbf_valid_predicted = [94.02000000000001, 96.179999999999993, 96.899999999999991, 97.099999999999994,
                            97.230000000000004, 97.310000000000002, 97.379999999999995, 97.359999999999999,
                            97.390000000000001, 97.359999999999999, 97.409999999999997]
    _rbf_test_predicted = [94.420000000000002, 96.099999999999994, 96.670000000000002, 97.040000000000006,
                           97.189999999999998, 97.189999999999998, 97.159999999999997, 97.260000000000005,
                           97.330000000000013, 97.340000000000003, 97.399999999999991]

    # plt.figure(figsize=[12,6])
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.title('SVM  performance (RBF kernel)')
    plt.plot(_cvalues, _rbf_train_predicted, label='Train data');
    plt.legend(loc='best')
    plt.plot(_cvalues, _rbf_valid_predicted, label='Validation data')
    plt.legend(loc='best')
    plt.plot(_cvalues, _rbf_test_predicted, label='Test data')
    plt.legend(loc='best')
    plt.show()

    linear_res = [97.286, 93.64, 93.78]
    rbf_1_res = [100.0, 15.48, 17.14]
    rbf_res = [94.294, 94.02, 94.42]

    plt.xlabel('Kernel Configurations')
    x = ['Train data', 'Validation data', 'Test data']
    plt.plot(linear_res, label='Linear')
    plt.plot(rbf_1_res, label='RBF (gamma=1)');
    plt.plot(rbf_res, label='RBF (default)');
    plt.xticks(range(len(x)), x)
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')
    plt.title('Comparison between SVM kernels using different parameters')
    plt.show()


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')

# Plot-only function
#plot()

print('\nLinear kernel');
#Linear kernel
clf = svm.SVC(kernel='linear')
clf.fit(train_data,np.ravel(train_label))
predicted_label= clf.predict(train_data)
print('\n Training set Accuracy:',accuracy_score(train_label,predicted_label)*100,'%')
predicted_label= clf.predict(validation_data)
print('\n Validation set Accuracy:',accuracy_score(validation_label,predicted_label)*100,'%')
predicted_label= clf.predict(test_data)
print('\n Test set Accuracy:',accuracy_score(test_label,predicted_label)*100,'%')

print('\nRBF with gamma = 1');
#RBF with gamma = 1
clf = svm.SVC(gamma=1)
clf.fit(train_data,np.ravel(train_label))
predicted_label= clf.predict(train_data)
print('\n Training set Accuracy:',accuracy_score(train_label,predicted_label)*100,'%')
predicted_label= clf.predict(validation_data)
print('\n Validation set Accuracy:',accuracy_score(validation_label,predicted_label)*100,'%')
predicted_label= clf.predict(test_data)
print('\n Test set Accuracy:',accuracy_score(test_label,predicted_label)*100,'%')

print('Default RBF\n');
#Default RBF
clf = svm.SVC()
clf.fit(train_data,np.ravel(train_label))
predicted_label= clf.predict(train_data)
print('\n Training set Accuracy:',accuracy_score(train_label,predicted_label)*100,'%')
predicted_label= clf.predict(validation_data)
print('\n Validation set Accuracy:',accuracy_score(validation_label,predicted_label)*100,'%')
predicted_label= clf.predict(test_data)
print('\n Test set Accuracy:',accuracy_score(test_label,predicted_label)*100,'%')


print('\nRBF Default with C varying form 1,10,20..100');
#RBF default, C from 1,10..100:
cvalues = [1,10,20,30,40,50,60,70,80,90,100]
rbf_train_predicted = []
rbf_valid_predicted = []
rbf_test_predicted = []
for i in cvalues:
    print('\n C=',i)
    clf = svm.SVC(C=i)
    clf.fit(train_data, np.ravel(train_label))
    predicted_label = clf.predict(train_data)
    acc = accuracy_score(train_label, predicted_label) * 100
    rbf_train_predicted.extend([acc])
    print('\n Training set Accuracy:', acc, '%')

    predicted_label = clf.predict(validation_data)
    acc = accuracy_score(validation_label, predicted_label) * 100
    rbf_valid_predicted.extend([acc])
    print('\n Validation set Accuracy:', acc, '%')

    predicted_label = clf.predict(test_data)
    acc = accuracy_score(test_label, predicted_label) * 100
    rbf_test_predicted.extend([acc])
    print('\n Test set Accuracy:', acc, '%')

print('Train accuracies:', rbf_train_predicted)
print('Validation accuracies:', rbf_valid_predicted)
print('Test accuracies:', rbf_test_predicted)

"""
Script for Extra Credit Part
"""
print('\nExtra credit\n')
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
