# 手动建立全连接神经网络\密集神经网络
# 作者：Jason（Shaoming，Wang）

# 方法一：面向对象

# 定义图上所有节点的父类节点
# 此节点具备向前传播、向后传播两种方法，子节点会在后面覆写
# forward propagation
# backward propagation
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
    """
    线性组合
    """
    def __init__(self,input_nodes, weights, bias):
        super(Node,self).__init__([input_nodes, weights, bias])

    def forward(self):
        input_values = self.inputs[0].value
        weight_values = self.inputs[1].value
        bias_values = self.inputs[2].value
        self.value = np.dot(input_values,weight_values) + bias_values

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
    """
    均方误差方法
    """
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


def toplogical_sort(feed_dict):
    """
    拓扑排序，需要完成的条件排在前面，使用Kahn's 算法
    :param feed_dict: 一个字典，键值是node，值是初始值
    :return:
    一个列表，记录了节点运行顺序
    """
    input_nodes = list(feed_dict.keys())

    # G是生成的graph
    G={}
    # 需要生成一个备份用于操作输入节点
    nodes = input_nodes.copy()

    # 以下代码是构建G的过程
    while len(nodes) > 0:
        node = nodes.pop(0)
        if node not in G:
            G[node] = {'in':set(), 'out':set()}
        for m in node.outputs:
            if m not in G:
                G[m] = {'in':set(), 'out':set()}
            G[m]['in'].add(node)
            G[node]['out'].add(m)
            nodes.append(m)

    # 初始化结果列表
    L = []
    S = set(input_nodes)

    while len(S)>0:
        node = S.pop()
        # 若此节点为输入节点实例，直接用初始值赋值
        if isinstance(node, Input_node):
            node.value = feed_dict[node]

        L.append(node)
        for m in node.outputs:
            # 判断是否所有条件都已经具备
            G[m]['in'].remove(node)
            G[node]['out'].remove(m)
            if len(G[m]['in']) == 0 :
                S.add(m)

    return L

def sgd_update(trainables, learn_rate=1e-2):
    """

    :param trainables: 需要训练的参数,是节点
    :param learn_rate: 学习率
    :return:
    """
    for t in trainables:
        t.value += -1 * learn_rate * t.gradients[t]

# 主程序
if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.utils import resample, shuffle

    # 数据导入
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # 标准化
    X_ = (X_ - np.mean(X_, axis=0))/ np.std(X_, axis=0)

    # 初始化参数
    n_feature = X_.shape[0]
    n_hidden = 10
    W1_ = np.random.randn(n_feature, n_hidden)
    B1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden,1)
    B2_ = np.zeros(1)

    X, y = Input_node(), Input_node()
    W1, b1 = Input_node(), Input_node()
    W2, b2 = Input_node(), Input_node()

    l1 = Linear_node(X, W1, b1)
    s1 = Activiation(l1)
    l2 = Linear_node(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X:X_,
        y:y_,
        W1:W1_,
        W2:W2_,
        b1:B1_,
        b2:B2_
    }

    epoch = 5000
    # 总共样本观测数
    observations_num = X_.shape[0]
    # batch数目
    batch_num = 16
    step_per_epoch = observations_num//batch_num

    # 经过拓扑排序，使得每一步的条件都具备后
    graph = toplogical_sort(feed_dict)
    trainables = [W1, W2, b1, b2]

    print('the total number of observations is ()'.format(observations_num))

    for i in range(epoch):
        loss =0
        # 这里不能设置random_state 参数，
        # 否则每次取得样本都一样了
        x_batch, y_batch =resample(X_, y_, n_samples=observations_num)

        sgd_update(trainables)


