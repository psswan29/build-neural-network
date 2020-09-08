# 手动建立全连接神经网络
# 作者：Jason


# 方法一：面向对象

# 定义图上所有节点的父类节点
# 此节点具备向前传播、向后传播两种方法，子节点会在后面覆写
# forward porpagation
# bakcward porpagation
# 否则会报一个NotImplementedError

# 同时定义多种属性 attributes
# inputs 输入对象，用一个列表容器呈装
# outputs 输出对象，用一个列表容器呈装
# value 是输出值，输出结果
# gradient 是梯度容器，是一个字典，要保存每次迭代结果

# 定义父节点
class Node(object):
    def __init__(self, inputs =[]):
        # 这两个属性记录每个节点连接的节点
        self.inputs = inputs
        self.outputs = []

        self.value = None
        self.gradient = {}
        # 如果每一个节点的输入链接到上一个节点输出，起连接点的作用
        # 若为起始输入点则
        for n in self.inputs:
            n.outpus.append(self)


    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class Input_node(Node):
    def __init__(self):
        super(Node,self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass

class Linear_node(Node):
    def __init__(self):
        super(Node,self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass

class Activiation(Node):
    def __init__(self):
        super(Node,self).__init__()
    def forward(self):
        pass
    def backward(self):
        pass


