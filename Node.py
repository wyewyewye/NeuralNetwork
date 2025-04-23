import random

# Perceptron 感知器
class Node:
    def __init__(self, inputs, activator):
        self._w = []
        for i in range(len(inputs)):
            self._w.append(random.random())
        self._bias = 0
        self._activator = activator

    def __str__(self):
        return f'w: {self._w}, bias: {self._bias}'

    def predict(self, inputs):
        if len(inputs) != len(self._w):
            raise Exception("The number of inputs is not equal to the number of weights")
        ans = 0.0
        for x, w in zip(inputs, self._w):
            ans += x * w
        ans += self._bias
        
        return self._activator(ans)
        
    def train(self, input_vec, labels, rate, iterations):
        for _ in range(iterations):
            self._train_once(input_vec, labels, rate)
        
    def _train_once(self, input_vec, labels, rate):
        samples = zip(input_vec, labels)
        for inputs, label in samples:
            output = self.predict(inputs)
            
    def _update_weights(self, input_vec, output, label, rate):
        error = output - label
    
    
