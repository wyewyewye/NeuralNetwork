

class LossFunc:
    def __init__(self, loss_func, loss_func_gradient):
        self.loss_func = loss_func
        self.loss_func_gradient = loss_func_gradient
        
    def loss_func(self, output, label):
        return self.loss_func(output, label)
    
    def loss_func_gradient(self, output, label):
        return self.loss_func_gradient(output, label)