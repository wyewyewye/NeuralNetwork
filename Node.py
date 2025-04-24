import random
from Activator import Activator
from LossFunc import LossFunc


# Perceptron 感知器
class Node:
    def __init__(self, inputs, activator: Activator, loss_func_obj: LossFunc):
        self._w = []
        for i in range(len(inputs)):
            # self._w.append(random.random())
            self._w.append(0.0)
        self._bias = 0.0
        self._activator = activator
        self._loss_func_obj = loss_func_obj

    def __str__(self):
        return f'w: {self._w}, bias: {self._bias}'

    def predict(self, inputs):
        if len(inputs) != len(self._w):
            raise Exception("The number of inputs is not equal to the number of weights")
        ans = 0.0
        for x, w in zip(inputs, self._w):
            ans += x * w
        ans += self._bias
        
        return self._activator.forward(ans)
        
    def train(self, inputs_vec, labels, rate, iterations):
        for _ in range(iterations):
            self._train_once(inputs_vec, labels, rate)
        
    def _train_once(self, inputs_vec, labels, rate):
        samples = zip(inputs_vec, labels)
        for inputs, label in samples:
            output = self.predict(inputs)
            self._update_weights(inputs, output, label, rate)
            
    def _update_weights(self, inputs, output, label, rate):
        # delta = (output - label) * self._activator.backward(output)
        # delta = (output - label) * 1(如果是阶跃函数)
        delta = self._loss_func_obj.loss_func_gradient(output, label) * self._activator.backward(output)
        new_w = []
        for x, w in zip(inputs, self._w):
            # 反向传播推导
            # w += rate * self._loss_func_obj.loss_func_gradient(output, label) * self._activator.backward(output) * x
            w -= rate * delta * x
            new_w.append(w)
        self._w = new_w
        
        self._bias -= rate * delta