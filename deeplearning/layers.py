import numpy as np

class Affine:
    """全连接层
    
    """
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
        self.x = None
        
        self.dw = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
    
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
    
class Softmax:
    def __init__(self, dim):
        self.dim = dim
    
    def forward(self, x):
        c = np.max(x, self.dim)
        epx_x = np.exp(x - c)
        sum_epx_x = np.sum(epx_x, axis=self.dim)
        y = epx_x / sum_epx_x
        
        return y
    
    def backward(self):
        pass