import numpy as np

# 以下layers支持batch， SoftmaxWithLoss 最后的计算取的是平均损失

class Affine:
    """全连接层
    
    """
    def __init__(self, input_size, out_size):
        self.w = np.random.randn(input_size, out_size)
        self.b = np.random.randn(out_size)
        
        self.x = None
        
        self.dw = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        
        # 雅可比矩阵
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

class SoftmaxWithLoss:
    def __init__(self, dim):
        self.dim = dim
    
    def forward(self, x):
        # Softmax
        c = np.max(x, self.dim)
        epx_x = np.exp(x - c)
        sum_epx_x = np.sum(epx_x, axis=self.dim)
        y = epx_x / sum_epx_x
        
        return y
    
    def backward(self, dout):
        
        dx = (self.y - self.t) * dout
        
        return dx
        