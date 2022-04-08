import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt

class MLP:
    " Multi-layer perceptron "

    def __init__(self, sizes, beta=1, momentum=0.9):

        """
        sizes is a list of length four. The first element is the number of features
                in each samples. In the MNIST dataset, this is 784 (28*28). The second
                and the third  elements are the number of neurons in the firs
                and the second hidden layers, respectively. The fourth element is the
                number of neurons in the output layer which is determined by the number
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means
                the first hidden layer has 5 neurons and the second layer has 7 neurons.

        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum
        """
        self.beta = beta
        self.momentum = momentum

        self.nin = sizes[0]  # number of features in each sample
        self.nhidden1 = sizes[1]  # number of neurons in the first hidden layer
        self.nhidden2 = sizes[2]  # number of neurons in the second hidden layer
        self.nout = sizes[3]  # number of classes / the number of neurons in the output layer

        # Initialise the network of two hidden layers
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden1) - 0.5) * 2 / np.sqrt(self.nin)  # hidden layer 1
        self.weights2 = (np.random.rand(self.nhidden1 + 1, self.nhidden2) - 0.5) * 2 / np.sqrt(
            self.nhidden1)  # hidden layer 2
        self.weights3 = (np.random.rand(self.nhidden2 + 1, self.nout) - 0.5) * 2 / np.sqrt(
            self.nhidden2)  # output layer

    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization
        niterations is the number of iterations for updating the weights

        """
        ndata = np.shape(inputs)[0]  # number of data samples
        # adding the bias
        inputs = np.concatenate((inputs, -np.ones((ndata, 1))), axis=1)

        # numpy array to store the update weights
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))

        for n in range(niterations):

            # forward phase
            self.outputs = self.forwardPass(inputs)

            # Error using the sum-of-squares error function
            error = 0.5 * np.sum((self.outputs - targets) ** 2)

            if (np.mod(n, 100) == 0):
                print("Iteration: ", n, " Error: ", error)

            # backward phase
            deltao = (self.outputs-targets)/self.outputs[0].shape*self.outputs*(1-self.outputs)

            # # compute the derivative of the second hidden layer
            err_layor2 = deltao.dot(self.weights3.T)
            deltah2 = np.delete(err_layor2 * self.sigmoid_prime(self.beta,self.input_hidden2),-1,axis=-1)

            # compute the derivative of the first hidden layer
            err_layor1 = deltah2.dot(self.weights2.T)
            deltah1 = np.delete(err_layor1 * self.sigmoid_prime(self.beta,self.input_hidden1),-1,axis=-1)

            # update the weights of the three layers: self.weights1, self.weights2 and self.weights3
            # here you can update the weights as we did in the week 4 lab (using gradient descent)
            # but you can also add the momentum
            updatew1 = inputs.T.dot(deltah1)*eta
            updatew2 = self.hidden1.T.dot(deltah2)*eta
            updatew3 = self.hidden2.T.dot(deltao)*eta

            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3

    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.
        """

        # layer 1
        # compute the forward pass on the first hidden layer with the sigmoid function
        self.hidden1 = np.zeros([inputs.shape[0],self.nhidden1])
        self.hidden1 = self.sigmoid(self.beta,np.dot(inputs,self.weights1))

        # layer 2
        # compute the forward pass on the second hidden layer with the sigmoid function
        input1 = inputs.dot(self.weights1)
        ndata1 = np.shape(self.hidden1)[0]
        self.input_hidden1 = np.concatenate((input1, np.ones((ndata1, 1))), axis=1)
        self.hidden1 = np.concatenate((self.hidden1, -np.ones((ndata1, 1))), axis=1)
        self.hidden2 = np.zeros([self.hidden1.shape[0],self.nhidden2])
        self.hidden2 = self.sigmoid(self.beta,np.dot(self.hidden1,self.weights2))

        # output layer
        # compute the forward pass on the output layer with softmax function
        input2 = self.hidden1.dot(self.weights2)
        ndata2 = np.shape(self.hidden2)[0]
        self.input_hidden2 = np.concatenate((input2, np.ones((ndata2, 1))), axis=1)
        self.hidden2 = np.concatenate((self.hidden2, -np.ones((ndata2, 1))), axis=1)
        outputs = np.zeros([self.hidden2.shape[0],self.nout])
        outputs = self.softmax(np.dot(self.hidden2,self.weights3))

        return outputs

    def evaluate(self, X, y):
        """
            this method is to evaluate our model on unseen samples
            it computes the confusion matrix and the accuracy

            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D.
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """
        inputs = np.concatenate((X, -np.ones((np.shape(X)[0], 1))), axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs, 1)
        targets = np.argmax(y, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("The confusion matrix is:")
        print(cm)
        print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)

        return cm

    def test_single(self,X):
        new_array = np.zeros((1, 785))
        for index in range(783):
            new_array[0][index] = X[index]
        # inputs = np.concatenate((X, -np.ones((np.shape(X)[0], 1))))
        new_array[0][784] = 0.0
        outputs = self.forwardPass(new_array)
        outputs = np.argmax(outputs, 1)
        return outputs[0]

    def sigmoid(self,beta,x):
        return 1/(1+np.exp(-beta*x))

    def sigmoid_prime(self,beta,x):
        return beta*self.sigmoid(beta,x)*(1-self.sigmoid(beta,x))

    def softmax(self, x):
        row_max = np.max(x, axis=1).reshape(-1, 1)
        x -= row_max
        x_exp = np.exp(x)
        s = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        return s

def run():
    f = gzip.open('mnist.pkl.gz', 'rb')
    tset, vset, teset = pickle.load(f, encoding='latin1')
    print(tset[0].shape, vset[0].shape, teset[0].shape)
    f.close()

    # Just use the first 9000 images for training
    tread = 9000
    train_in = tset[0][:tread, :]
    # This is a little bit of work -- 1 of N encoding
    # Make sure you understand how it does it
    train_tgt = np.zeros((tread, 10))
    for i in range(tread):
        train_tgt[i, tset[1][i]] = 1

    # and use 1000 images for testing
    teread = 1000
    test_in = teset[0][:teread, :]
    test_tgt = np.zeros((teread, 10))
    for i in range(teread):
        test_tgt[i, teset[1][i]] = 1
    sizes = [784, 32, 64, 10]  # 784 is the number of pixels of the images and 10 is the number of classes
    classifier = MLP(sizes)
    classifier.train(train_in, train_tgt, 0.01, 1000)
    classifier.evaluate(test_in, test_tgt)


if __name__ == "__main__":
    run()