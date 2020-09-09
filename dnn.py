# 手动建立全连接神经网络
# 作者：Jason（shaoming，wang）

# 方法一：面向对象

# 定义图上所有节点的父类节点
# 此节点具备向前传播、向后传播两种方法，子节点会在后面覆写
# forward porpagation
# bakcward porpagation
# 否则会报一个NotImplementedError

# 同时定义多种属性 attributes
# inputs以及outputs是用来构建一个网络graph
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
    # todo
    def backward(self):
        # 为所有网络中传入节点梯度初始化
        # 记录了所有传入节点的梯度信息
        self.gradients = {node:np.zeros_like(node.value) for node in self.inputs}

        for node in self.outputs:
            gradients_cost = node.gradients[self]
            self.gradients[self.inputs[0]] = np.dot(gradients_cost,self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, gradients_cost)
            self.gradients[self.inputs[2]] = np.sum(gradients_cost, axis=0, keepdims=False)
        # XW + B / W ==> X
        # XW + B / X ==> W

class Activiation(Node):
    # 激活函数
    def __init__(self, node, act_type='sigmoid'):
        super(Node,self).__init__([node])
        self.act_type = act_type
    # 向前传播
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        if self.act_type == 'sigmoid':
            self.value = self._sigmoid(self.x)

    def backward(self):
        # y = 1 / (1 + e^-x)
        # y' = 1 / (1 + e^-x) (1 - 1 / (1 + e^-x))
        if self.act_type == 'sigmoid':
            self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        # 参数初始化
        self.gradients = {node:np.zeros_like(node.value) for node in self.inputs}

        for node in self.outputs:
            grad_cost = node.gradients[self]
            self.gradients[self.inputs[0]]= grad_cost * self.partial

class MSE(Node):
    def __init__(self, y_truth, node):
        super(Node,self).__init__([y_truth, node])

    def forward(self):
        y_truth = self.inputs[0]
        y_pred = self.inputs[1].value

        # 为了不出错，保证真值与预测值之间的shape是相同的
        # 利用属性进行变量的传递
        y_truth = y_truth.reshape(-1,1)
        self.m = y_truth.shape[0]
        y_pred = y_pred.reshape(-1,1)
        assert y_truth.shape == y_pred.shape
        self.diff = y_truth-y_pred
        self.value = np.mean((self.diff)**2)

    def backward(self):
        # 这里是对损失函数求偏微分
        self.gradients[self.inputs[0]] = 2 * np.mean(self.diff)
        self.gradients[self.inputs[1]] = -2 * np.mean(self.diff)

#   todo: 完成一个拓扑排序
def toplogical_sort():
    pass

if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.utils import shuffle, resample

    # 数据导入
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # 标准化
    X_ = (X_ - np.mean(X_, axis=0))/ np.std(X_, axis=0)






