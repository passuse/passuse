import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        """One updating step, update weights"""

        layer = self.model
        if layer.trainable:

            ###########################################################################
            # TODO: Put your code here
            # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
            # You need to add momentum to this.

            # Weight update with momentum
            # Update the velocity for W and b using the gradient and the momentum
            self.v_m=self.momentum*self.v_m- self.learning_rate * layer.grad_W
            self.v_b = self.momentum * self.v_b - self.learning_rate * layer.grad_b
            # Update the weight and bias using the velocity
            layer.W += self.v_W
            layer.b += self.v_b
            # # Weight update without momentum
            # layer.W += -self.learning_rate * layer.grad_W
            # layer.b += -self.learning_rate * layer.grad_b

            ############################################################################
