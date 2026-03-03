from collections import OrderedDict
import numpy as np
from layers import *

class TwoLayerNet:

    def __init__(self, input_size=28 * 28, hidden_size=1024 * 1024, out_size=10, lr=0.01):
        # 训练参数
        self.lr = lr
        
        # 临时数据
        self.y = None
        self.t = None
        
        # 网络构建，使用OrderedDict模拟nn.Sequential
        self.layers = OrderedDict()
        self.layers['layer1'] = Affine(input_size, hidden_size)
        self.layers['ReLU1'] = ReLU()
        self.layers['layer2'] = Affine(hidden_size, out_size)
        self.layers['ReLU2'] = ReLU()
        self.layers['LastLayer'] = SoftmaxWithLoss(dim=1)
        
    # 模型权重
    @property
    def params(self):
        return {
            'layer1': {
                'w': self.layers['layer1'].w, 'b': self.layers['layer1'].b
            },
            'layer2': {
                'w': self.layers['layer2'].w, 'b': self.layers['layer2'].b
            }
        }
    
    # 向前传播
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    # 计算损失
    def loss_fn(self, y, t):
        self.y = y
        self.t = t
        
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
            
        batch_size = y.shape[0]
        loss = -np.sum(t * np.log(y + 1e-7)) / batch_size
        
        return loss
    
    # 向后求梯度
    def backward(self):
        batch_size = self.y.shape[0]
        
        dout = 1.0 / batch_size
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
    
    # 更新权重
    def step(self):
        for layer in self.layers.values():
            if(layer.w and layer.b):
                layer.w -= layer.dw * self.lr
                layer.b -= layer.db * self.lr
                
    def save(self, path):
        save_dict = {}
        
        for layer_name, layer_params in self.params.items():
            for param_name, value in layer_params.items():
                key = f'{layer_name}_{param_name}'
                save_dict[key] = value
                
        np.savez(path, **save_dict)
        
    def load(self, path):
        data = np.load(path)
        
        for layer_name, layer_params in self.params.items():
            for param_name in layer_params.keys():
                key = f"{layer_name}_{param_name}"
                if key in data:
                    param = data[key]
                    if layer_name in self.layers:
                        if param_name == 'w':
                            self.layers[layer_name].w = param
                        elif param_name == 'b':
                            self.layers[layer_name].b = param