import os,sys,time
import numpy as np
import matplotlib.pyplot as plt
import mnist



def load_dataset():
    print(f'\n Loading dataset...\n')
    mndata = mnist.MNIST('./data/')
    Xtrain, Ytrain = map(np.array, mndata.load_training())
    X_test, Y_test = map(np.array, mndata.load_testing())
    Xtrain = Xtrain/255.0
    X_test = X_test/255.0
    return Xtrain,Ytrain,X_test,Y_test

def split_dataset(X,Y,p):
    print(f'\n Splitting dataset...')
    n,d = X.shape
    Xtrn,Ytrn = X[np.arange(np.int(n*p))],Y[np.arange(np.int(n*p))]
    Xval,Yval = X[np.arange(np.int(n*p),n-1)],Y[np.arange(np.int(n*p),n-1)]
    return Xtrn,Ytrn,Xval,Yval

def permute(X):
    n,d = X.shape
    idxs = np.arange(n)
    ridx = np.random.permutation(idxs)
    return X[ridx], ridx



def display_digit(flat_digit,label):
    print("\n Displaying image. Close to continue...")
    plt.figure(figsize=plt.figaspect(1.0))
    plt.subplot(1,1,1)
    plt.imshow(flat_digit.reshape([28,28]), cmap=plt.cm.gray)
    plt.title(f'Random MNIST Digit: {label}', fontsize=20)
    plt.show()

def plot_errors(p,trn,tst,tms):
    plt.figure(figsize=plt.figaspect(0.5))
    plt.style.use('seaborn-whitegrid')
    plt.title(f'Errors w.r.t. Increasing P',fontsize=16,fontweight='bold')
    plt.xlabel(f'P',fontsize=12,fontweight='bold')
    plt.ylabel(f'Errors',fontsize=12,fontweight='bold')
    plt.plot(p,trn.T,'-',color='#58e',label='Training Error')
    plt.plot(p,tst.T,'-',color='#9f9',label='Validation Error')
    plt.plot(p,tms.T,'-',color='#f99',label='Training Times (s)')
    plt.legend(loc="upper right",frameon=True,borderpad=1,borderaxespad=1,facecolor='#fff',edgecolor='#777',shadow=True)
    plt.show()



class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.params = self.initialization() # save parameters in dictionary

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))



if __name__ == "__main__":
	dnn = DeepNeuralNetwork([[32,16],[16,16],[16,10]])
