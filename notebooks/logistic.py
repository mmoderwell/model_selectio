'''
@author: Matt
@description: An implementation of Logistic Regression
'''

import numpy as np

def sigmoid(z, derivative=False):
    sigm = 1.0 / (1.0 + np.exp(-z))
    if derivative:
        return sigm * (1.0 - sigm)
    return sigm

def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

#mean absolute error
def mean_absolute(features, labels, weights):
    observations = len(labels)
    predictions = predict(features, weights)
    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)
    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)
    #Take the sum of both costs
    cost = class1_cost - class2_cost
    #Take the average cost
    cost = cost.sum()/observations
    return cost

#least squares cost function
def least_squares(features, labels, weights):
    predictions = predict(features, weights)
    #Take the error when label=1
    cost = np.square(predictions - labels)
    #Take the average cost
    cost = cost.sum() / len(labels)
    return cost

# vectorized gradient descent
def update_weights(features, labels, weights, alpha):
    N = len(features)
    #get predictions
    predictions = predict(features, weights)
    # Returns a matrix holding a partial derivative, one for each feature
    gradient = np.dot(features.T,  predictions - labels)
    #take the average cost derivative for each feature
    gradient /= N
    #multiply the gradient by learning rate
    gradient *= alpha
    #subtract from the weights to minimize cost
    weights -= gradient
    return weights

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

def classify(predictions):
    decision = np.vectorize(decision_boundary)
    return decision(predictions).flatten()

def confusion_matrix(predicted, actual):
    true_pos, true_neg ,false_pos, false_neg = 0, 0, 0, 0
        
    for i in range(len(predicted)):
        if predicted[i] == actual[i] and predicted[i] == 1:
            true_pos += 1
        elif predicted[i] == actual[i] and predicted[i] == 0:
            true_neg += 1
        elif predicted[i] != actual[i] and predicted[i] == 1:
            false_pos += 1
        elif predicted[i] != actual[i] and predicted[i] == 0:
            false_neg += 1
    
    return true_pos, true_neg, false_pos, false_neg

def train(features, labels, alpha, iters, verbose=True):
    weights = 0.1 * np.random.rand(features.shape[1]).reshape(features.shape[1], 1)
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, alpha)
        #Calculate error for auditing purposes
        cost = least_squares(features, labels, weights)
        cost_history.append(cost)
        if i % 25 == 0 and verbose:
            print(i, ':', cost)
    if verbose:
        print ('Training complete after', iters, 'iterations.')
        print ('Cost: ', cost)
    return weights, cost_history

def plot_cost_history(ch):
    plt.figure(figsize=(10,6), frameon=False)
    # plot the points as scatter plot
    plt.scatter(range(len(ch)), ch, color = "gray", marker = "o", s = 10)
    # add the labels
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    # show the plot
    plt.show()