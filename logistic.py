__author__ = 'dumengnan'

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

plot_decision_boundary(lambda x: clf.predict(x))
plt.title('logistic regression')
plt.show()
num_examples = len(X)
alpha = 0.01  # learning rate
reg_lambda = 0.01

def calculate_loss(theta):
    X_expand = np.zeros((X.shape[0], X.shape[1]+1))
    X_expand[:,:-1]=X
    X_expand[:,-1:]=1
    z = X_expand.dot(theta)
    probs = sigmoid(z)
    data_loss = np.power((probs-y.reshape(X.shape[0],1)), 2)
    data_loss = np.sum(data_loss)
    return 1.0/num_examples*data_loss

def predict(theta, x):
    x_expand = np.zeros((x.shape[0], x.shape[1]+1))
    x_expand[:,:-1] = x
    x_expand[:,-1:] = 1
    z = x_expand.dot(theta)
    probs = sigmoid(z)

    return np.array([int(item>0.5) for item in probs])


def sigmoid(x):
    return (1+np.tanh(x/2.0))/2.0


def build_model(num_passes=200, print_loss=False):
    X_expand = np.zeros((X.shape[0], X.shape[1]+1))
    X_expand[:,:-1]=X
    X_expand[:,-1:]=1
    ndim = X_expand.shape[1]
    np.random.seed(0)
    theta = np.random.randn(ndim, 1)/ndim

    for i in xrange(0, num_passes):

        # batch gradient descent
        delta_theta = X_expand.T.dot(sigmoid(X_expand.dot(theta))-y.reshape(X.shape[0],1))
        delta_theta += reg_lambda*theta

        # gradient descent parameter update
        theta += -alpha*delta_theta

        if print_loss and i%2 == 0:
            print "loss after iteration %i: %f" %(i, calculate_loss(theta))

    return theta

theta = build_model(print_loss=True)
plot_decision_boundary(lambda x:predict(theta, x))
plt.title('logistic regression with batch gradient descent')
plt.show()