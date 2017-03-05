import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from datetime import datetime
from time import sleep
import math

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1 + np.exp(-z))


def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('../mnist_all.mat')  # loads the MAT object as a Dictionary
    # for i in mat2:
    #     print(i)
    #     for y in mat2[i]:
    #         print(y)
    #         sleep(5)

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    w1_transpose = np.transpose(w1)
    w2_transpose = np.transpose(w2)

    # Check
    training_data = np.append(training_data, np.ones((training_data.shape[0],1)), 1)
    #print("Should actually be 1 : ", training_data[20000, -1])

    zj = np.dot(training_data, w1_transpose)

    zj = sigmoid(zj)

    zj = np.append(zj, np.ones((zj.shape[0], 1)),1)

    ol = np.dot(zj, w2_transpose)

    ol = sigmoid(ol)

    ol_log = np.log(ol)

    label_mod = np.zeros((training_data.shape[0],n_class))

    for i in range(training_label.shape[0]):
        index = int(training_label[i])
        # print(index)
        label_mod[i][index] = 1

    # print("Y and Log")
    #print(label_mod.shape)
    #print(ol_log.shape)

    part1 = np.multiply(label_mod, ol_log)
    part2a = np.subtract(1,label_mod)
    part2b = np.log(np.subtract(1, ol))

    finalSolution_parta = np.add(part1, np.multiply(part2a, part2b))

    finalSolution_parta = np.divide(np.sum(finalSolution_parta),(-1)*training_data.shape[0])

    # print("NEXT")

    finalSolution_partb = (np.sum(np.square(w1)) + np.sum(np.square(w2))) * np.divide(lambdaval,(2*training_data.shape[0]))

    obj_val = finalSolution_parta + finalSolution_partb

    # print("Reached here")

    #Calculate grad descent
    #Calculate w2
    derivate2 = np.dot(np.transpose(np.subtract(ol,label_mod)), zj)
    # print("derivate2",derivate2.shape)
    # print("matrix1",sum2.shape)
    grad_w2 = np.divide(np.add(derivate2,np.multiply(lambdaval,w2)),training_data.shape[0])
    # print("grad_w2",grad_w2.shape)


    #calculate grad_w1
    modified_w2 = w2[:,0:w2.shape[1]-1]
 #   print("modified_w2",modified_w2.shape)
    t1 = np.dot(np.subtract(ol,label_mod),modified_w2)
  #  print("t1",t1.shape)
    # t2 = np.multiply(t1,training_data)
    zj = zj[:,0:zj.shape[1]-1]
    t3 = np.multiply(np.subtract(1,zj),zj)
    # print("t2" , t2.shape)
   # print("t3", t3.shape)
    t = np.multiply(t3,t1)
    grad_w1 = np.dot(t.T,training_data)
    # print("partasum",partasum.shape)
    grad_w1 = np.add(grad_w1,np.multiply(lambdaval,w1))/training_data.shape[0]
    #print("grad_w1",grad_w1.shape)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
   # print(obj_val)
    # print("Shape", obj_grad.shape)
    # print("Value", obj_val)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""
    # print("data", data.shape)
    # print("w1", w1.shape)
    # print("w2", w2.shape)
    data_with_bias = np.append(data, np.ones((data.shape[0],1)), 1)
    # print("data_with_bias", data_with_bias.shape)
    #Feed Forward
    zj = np.dot(data_with_bias, np.transpose(w1))

    zj = sigmoid(zj)

    zj = np.append(zj, np.ones((zj.shape[0],1)), 1)

    ol = np.dot(zj, np.transpose(w2))

    ol = sigmoid(ol)

    # print("ol",ol.shape)
    # labels = np.zeros((ol.shape[0],1))
    # #print("label" , labels.shape)
    # # Your code here
    # #max = 0
    # # print("ol print: ", ol)
    # # sleep(5)
    # for i in range(ol.shape[0]):
    #     m = np.argmax(ol[i])
    #     labels[i][0] = m
        # print ("Argmax", m)
    # print("labels shape", labels.shape)

    labels = np.argmax(ol,axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
# print("train label" , train_label)
# print("train label shape" , train_label.shape)
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

print('#Hidden\tLambda\tTrain Acc\tValidation Acc\tTest acc\n')
#best value of hidden layers
for j in range(20,101,10):  # 20 - 100 hidden layers
    for k in range(0,61,5): # 0 - 60    lambda
    # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = j

        # set the number of nodes in output unit
        n_class = 10

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter
        lambdaval = k

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        # start = datetime.now().replace(microsecond=0)
        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        # start = datetime.now().replace(microsecond=0) - start

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset

        # print(predicted_label.shape)
        # print(train_label.shape)
        # print('\nHidden layers:'+ str(n_hidden)+'\n')
        # print('\nLambda:'+str(lambdaval)+'\n')
        #
        # print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
        #
        # predicted_label = nnPredict(w1, w2, validation_data)
        #
        # # find the accuracy on Validation Dataset
        #
        # print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
        #
        # predicted_label = nnPredict(w1, w2, test_data)
        #
        # # find the accuracy on Validation Dataset
        #
        # print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%'+'\n')

        print(str(n_hidden),end='\t')
        print(str(lambdaval),end='\t')
        predicted_label = nnPredict(w1, w2, train_data)
        print(str(100*np.mean((predicted_label == train_label).astype(float))),end='\t')
        predicted_label = nnPredict(w1, w2, validation_data)
        print(str(100*np.mean((predicted_label == validation_label).astype(float))),end='\t')
        predicted_label = nnPredict(w1, w2, test_data)
        print(str(100*np.mean((predicted_label == test_label).astype(float))),end='\t')
        # print(str(start),end='\t')
        #print(str(n_hidden)+','+str(lambdaval))+','+str(100*np.mean((predicted_label == train_label).astype(float)))+','+str(100*np.mean((predicted_label == validation_label).astype(float)))+ ','+ str(100*np.mean((predicted_label == test_label).astype(float)))
        print('\n')