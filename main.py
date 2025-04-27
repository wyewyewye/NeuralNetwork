import math
from LossFunc import LossFunc
from Activator import Activator
from Perceptron import Perceptron

def MSE_loss(output, label):
    return 0.5 * (output - label) ** 2

def MSE_loss_gradient(output, label):
    return output - label

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def step(x):
    return 1 if x > 0 else 0

def step_gradient(x):
    # STE 直通估计器
    return 1

if __name__ == "__main__":
    MSE_loss_func = LossFunc(MSE_loss, MSE_loss_gradient)
    step_activator = Activator(step, step_gradient)
    sigmoid_activator = Activator(sigmoid, sigmoid_gradient)
    
    # data set
    inputs_vec = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 1]
    
    perceptron = Perceptron(inputs_vec[0], sigmoid_activator, MSE_loss_func)
    perceptron.train(inputs_vec, labels, 0.1, 10000)
    print(perceptron)
    
    # test
    for inputs in inputs_vec:
        print(f'{inputs} -> {perceptron.predict(inputs)}')