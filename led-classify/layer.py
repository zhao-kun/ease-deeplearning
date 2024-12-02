import numpy

# 层基类
class Layer:
    # 初始化
    def __init__(self):
        pass
    
    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

    

