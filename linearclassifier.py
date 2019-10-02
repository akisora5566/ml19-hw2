"""
Functions for training and predicting with linear classifiers
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize, check_grad


def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of features. 
    
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :return: length n vector of class predictions
    :rtype: array
    """
    # Predict the class by finding the highest scoring linear combination of features
    # Rewritten by akisora5566 on 9/27
    # Avoid the for loop. Use the matrix multiplication instead.

    d, num_classes = model["weights"].shape     #(2, 4): 2 features, 4 classess

    weights = model["weights"].transpose()
    weighted_scores = np.matmul(weights, data)          # (#classes, d) * (d, n) = (#classes, n)
    predict_class = np.argmax(weighted_scores, axis=0)  # (,n)

    return predict_class


def perceptron_update(data, model, label):
    """
    Update the model based on the perceptron update rules and return whether the perceptron was correct
    
    :param data: (d, 1) ndarray representing one example input
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param label: the class label of the single example
    :type label: int
    :return: whether the perceptron correctly predicted the provided true label of the example
    :rtype: bool
    """
    # Directly updating the model dict and returning the proper boolean value
    score_one_data = np.dot(model["weights"].T, data)
    predict = np.argmax(score_one_data, axis=0)
    if label != predict:
        model["weights"][:, label] = model["weights"][:, label] + data
        model["weights"][:, predict] = model["weights"][:, predict] - data
        return False
    else:
        return True


def log_reg_train(data, labels, params, model=None, check_gradient=False):
    """
    Train a linear classifier by maximizing the logistic likelihood (minimizing the negative log logistic likelihood)
     
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param labels: length n array of the integer class labels 
    :type labels: array
    :param params: dictionary containing 'lambda' key. Lambda is the regularization parameter and it should be a float
    :type params: dict
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param check_gradient: Boolean value indicating whether to run the numerical gradient check, which will skip
                            learning after checking the gradient on the initial model weights.
    :type check_gradient: Boolean
    :return: the learned model 
    :rtype: dict
    """
    d, n = data.shape
    num_classes = np.unique(labels).size

    if model:
        weights = model['weights'].ravel()
    else:
        weights = np.zeros(d * num_classes)

    def log_reg_nll(new_weights):
        """
        This internal function returns the negative log-likelihood (nll) of the data given the logistic regression weights
        
        :param new_weights: weights to use for computing logistic regression likelihood
        :type new_weights: ndarray
        :return: tuple containing (<negative log likelihood of data>, gradient)
        :rtype: tuple
        """
        # reshape the weights, which the optimizer prefers to be a vector, to the more convenient matrix form
        new_weights = new_weights.reshape((d, num_classes))

        # Compute the objective value (nll)

        w_x = np.matmul(new_weights.T, data)        #(num_classes,n)
        logsumexp_w_x = logsumexp(w_x, dim=0)       #(1,n)
        sum_logsumexp_w_x = np.sum(logsumexp_w_x)

        sum_wyi_xi = 0
        for i in range (0,n):
            wyi_xi = np.matmul(new_weights[:, labels[i]].T, data[:,i])
            sum_wyi_xi += wyi_xi

        frob_weights = np.linalg.norm(new_weights, ord='fro')

        nll = (params["lambda"] / 2) * np.square(frob_weights) + sum_logsumexp_w_x - sum_wyi_xi    # 1 number, not a matrix

        # Compute the gradient
        gradient = np.zeros((d, num_classes))
        for c in range (0, num_classes):
            wc_x = np.matmul(new_weights[:, c].T, data)     # (1,n)
            logged_first_term = wc_x - logsumexp_w_x        # (1,n)
            first_term = np.exp(logged_first_term)          # (1,n)
            # - I(yi == c)
            for i in range (0,n):
                if labels[i] == c:
                    first_term[0, i] = first_term[0, i] - 1
            term_in_sum = np.multiply(data, first_term)
            sum_term = np.sum(term_in_sum, axis=1)          # single constant
            gradient_wc = np.multiply(params["lambda"], new_weights[:,c]) + sum_term    # (d,)
            gradient[:, c] = gradient_wc

        return nll, gradient

    if check_gradient:
        grad_error = check_grad(lambda w: log_reg_nll(w)[0], lambda w: log_reg_nll(w)[1].ravel(), weights)
        print("Provided gradient differed from numerical approximation by %e (should be around 1e-3 or less)" % grad_error)
        return model

    # pass the internal objective function into the optimizer
    res = minimize(lambda w: log_reg_nll(w)[0], jac=lambda w: log_reg_nll(w)[1].ravel(), x0=weights)
    weights = res.x

    model = {'weights': weights.reshape((d, num_classes))}

    return model


def plot_predictions(data, labels, predictions):
    """
    Utility function to visualize 2d, 4-class data 
    
    :param data: 
    :type data: 
    :param labels: 
    :type labels: 
    :param predictions: 
    :type predictions: 
    :return: 
    :rtype: 
    """
    num_classes = np.unique(labels).size

    markers = ['x', 'o', '*',  'd']

    for i in range(num_classes):
        plt.plot(data[0, np.logical_and(labels == i, labels == predictions)],
                 data[1, np.logical_and(labels == i, labels == predictions)],
                 markers[i] + 'g')
        plt.plot(data[0, np.logical_and(labels == i, labels != predictions)],
                 data[1, np.logical_and(labels == i, labels != predictions)],
                 markers[i] + 'r')


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    
    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val
