import numpy as np


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, out_size):

        self.params = {}
        
        self.params['w1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        
        self.params['w2'] = np.random.randn(hidden_size, out_size)
        self.params['b2'] = np.random.randn(out_size)
        
    def predict(self, input):
        pass
    
    
