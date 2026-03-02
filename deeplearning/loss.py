import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass
    
    def forward(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size