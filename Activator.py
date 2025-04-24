
class Activator:
    def __init__(self, step_activator, step_activator_gradient):
        self.activator = step_activator
        self.activator_gradient = step_activator_gradient
        
    def forward(self, x):
        return self.activator(x)
    
    def backward(self, x):
        return self.activator_gradient(x)