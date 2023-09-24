import numpy as np
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11


class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        """
        Apply a linear transformation to the incoming data: y = Wx + b
        Args:
            num_input: size of each input sample
            num_output: size of each output sample
            trainable: whether if this layer is trainable
        """

        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
          Inputs: (minibatch)
          - Input: (batch_size, 784)
          - labels: the ground truth label, shape (batch_size, )
        """

        ############################################################################
        # TODO: Put your code here
        # Apply linear transformation (WX+b) to Input, and then
        # calculate the average accuracy and loss over the minibatch
        # Return the loss and acc, which will be used in solver.py
        # Hint: Maybe you need to save some arrays for gradient computing.

        ############################################################################
        # Apply linear transformation (WX+b) to Input
        scores = np.dot(Input, self.W) + self.b  # shape: (batch_size, 10)

        # Apply softmax function to scores
        exp_scores = np.exp(scores)  # shape: (batch_size, 10)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # shape: (batch_size, 10)

        # Calculate the softmax cross entropy loss
        correct_logprobs = -np.log(probs[range(batch_size), labels])  # shape: (batch_size,)
        loss = np.sum(correct_logprobs) / batch_size  # scalar

        # Calculate the accuracy
        preds = np.argmax(probs, axis=1)  # shape: (batch_size,)
        acc = np.mean(preds == labels)  # scalar

        # Save some arrays for gradient computing
        self.Input = Input  # shape: (batch_size, 784)
        self.probs = probs  # shape: (batch_size, 10)
        self.labels = labels  # shape: (batch_size,)

        return loss, acc
        return loss, acc

    def gradient_computing(self):
        ############################################################################
        # TODO: Put your code here
        # Calculate the gradient of W and b.

        # self.grad_W = 
        # self.grad_b =
        ############################################################################
        # Get the batch size
        batch_size = self.Input.shape[0]

        # Calculate the gradient of loss with respect to scores
        dscores = self.probs  # shape: (batch_size, 10)
        dscores[range(batch_size), self.labels] -= 1  # shape: (batch_size, 10)
        dscores /= batch_size  # shape: (batch_size, 10)

        # Calculate the gradient of scores with respect to W and b
        self.grad_W = np.dot(self.Input.T, dscores)  # shape: (784, 10)
        self.grad_b = np.sum(dscores, axis=0, keepdims=True)  # shape: (1, 10)

    def XavierInit(self):
        """
        Initialize the weigths
        """
        raw_std = (2 / (self.num_input + self.num_output)) ** 0.5
        init_std = raw_std * (2 ** 0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
