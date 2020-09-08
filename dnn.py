# 手动建立全连接神经网络
# 作者：Jason（shaoming，wang）

import numpy as np
# 方法一：面向对象

# 定义图上所有节点的父类节点
# 此节点具备向前传播、向后传播两种方法，子节点会在后面覆写
# forward porpagation
# bakcward porpagation
# 否则会报一个NotImplementedError

# 同时定义多种属性 attributes
# inputs 所有输入对象，用一个列表容器呈装
# outputs 所有输出对象，用一个列表容器呈装
# value 是输出值，输出结果
# gradient 是梯度容器，是一个字典，要保存每次迭代结果

# 定义父节点,这是神经网络中所有节点的父节点
class Node(object):
    def __init__(self, inputs =[]):
        # 这两个属性记录每个节点连接的节点
        self.inputs = inputs
        self.outputs = []

        self.value = None
        self.gradients = {}

        # 如果每一个节点的输入链接到上一个节点输出，起连接点的作用
        # 若为起始输入点则是每个节点的输出节点
        for n in self.inputs:
            n.outpus.append(self)


    def forward(self):
        # 向前传播，需要被覆盖不然会报错
        raise NotImplementedError
    def backward(self):
        # 反向传播，需要被override
        raise NotImplementedError

# 此类节点包括输入样本，权重weights，偏差bias
# bias 可以简化为一个参数，广播至前面 inputs * w + b
class Input_node(Node):
    # 因为输入节点不属于内部节点，可以不传递其他参数
    def __init__(self):
        super(Node,self).__init__()

    # 只有在输入节点, value 被传递给forward
    # 其他节点只能从前一个节点得到
    def forward(self,value=None):
        if value is not None:
            self.value = value

    def backward(self):
        # 梯度初始化
        self.gradients = {self:0}
        for node in self.outputs:
            gradients_cost = node.gradients[self]
            # 梯度的传递遵循链式法则，
            # 因此是一个连乘的形式
            self.gradients[self] = 1 * gradients_cost

class Linear_node(Node):

    def __init__(self,input_nodes, weights, bias):
        super(Node,self).__init__([input_nodes, weights, bias])

    def forward(self):
        input_values = self.inputs[0].value
        weight_values = self.inputs[1].value
        bias_values = self.inputs[2].value
        self.value = np.dot(input_values,weight_values) + bias_values
    def backward(self):
        # 为所有网络中传入节点梯度初始化
        self.gradients = {node:np.zeros_like(node.value) for node in self.inputs}
        for node in self.outputs:
            gradients_cost = node.gradients[self]
            self.gradients[self.inputs[0]] = np.dot()
            self.gradients[self.inputs[1]] = np.dot()
            self.gradients[self.inputs[2]]

class Activiation(Node):
    def __init__(self):
        super(Node,self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass

class MSE(Node):
    def __init__(self):
        super(Node,self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass

