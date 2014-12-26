#-*- coding: utf-8 -*-
from math import exp


def sigmoid(n):
    return 1.0 / (1 + exp(-n))


def dsigmoid(n):
    return sigmoid(n) * (1.0 - sigmoid(n))


class NetTable:
    
    # in 表示 input_hidden, ho 表示 hidden_output
    # hnn 表示 Hidden Node Number, 即隐层的节点数目
    ih, ho = [], []
    hnn, alpha = 0, 0

    def __init__(self, hnn_ = 20, alpha_ = 0.1):
        """ 初始化权重

        1. 会将输入层和隐藏层之间的连接权重初始化为：-0.2
        2. 隐藏层和输出层之间的连接权重初始化为：0

        :param hnn_: 设置隐层节点的数目, 建议值为：8~25

        """
        self.hnn = hnn_
        self.alpha = alpha_
        self.ih = [[0.2 for j in range(self.hnn)] for i in range(64)]
        self.ho = [[0.05 for k in range(10)] for j in range(self.hnn)]


    def get_weight(self, layer, fromid, toid):
        """ 查询两个节点之间的权重

        :param layer: 指定查询哪一层之间的节点，0代表输入层与隐层；1代表隐层与输出层
        :param fromid: 
            `当layer=0时`: 0 < fromid < 63, 0 < toid < hiddenNodeNum
            `当layer=1时`: 0 < fromid < hiddenNodeNum, 0 < toid < 10

        """
        target = self.ih if layer == 0 else self.ho
        weight = target[fromid][toid]

        return weight


    def update_weight(self, layer, fromid, toid, weight):

        target = self.ih if layer == 0 else self.ho
        target[fromid][toid] += weight


    def feed_forward(self, input):
        """ 信息正向传递,前馈查询

        :param record: 一个含64个数字的列表，代表一个数字的64维的特征向量
        :returns:
            `output_o` - 对应每一个隐藏层的输出
            `output_h` - 对应0~9的10个数字的可能性

        """

        # 两层的输出数组
        output_h = [1.0] * self.hnn
        output_o = [1.0] * 10

        # i, j , k分别是输入层，隐层，输出层的Node的下标

        # 计算隐藏层的输出结果
        for j in range(self.hnn):
            sum = 0.0
            for i in range(64):
                sum = sum + input[i] * self.ih[i][j]
            output_h[j] = round(sigmoid(sum), 4)

        # 计算输出层的输出结果
        for k in range(10):
            sum = 0.0
            for j in range(self.hnn):
                sum = sum + output_h[j] * self.ho[j][k]
            output_o[k] = round(sigmoid(sum), 4)

        return (output_h, output_o)

    
    def back_propagate(self, input, target):
        """ 误差后向传播调整权重

        :param input: 64维特征向量
        :param target: 一个与input相对应的整数

        """

        # 将targets中与正确数字对应的，值设为0.9，其余的设为0.1
        targets = [0.0] * 10
        targets[target] = 1.0

        # 获取将input推入网络后，每个节点的输出
        output_h, output_o = self.feed_forward(input)

        # 计算输出层误差
        deltas_o = [0.0] * 10
        for k in range(10):
            error = targets[k] - output_o[k]
            deltas_o[k] = dsigmoid(output_o[k]) * error

        # 计算隐藏层误差
        deltas_h = [0.0] * self.hnn
        for j in range(self.hnn):
            error = 0.0
            for k in range(10):
                error = error + deltas_o[k] * self.ho[j][k]
            deltas_h[j] = dsigmoid(output_h[j]) * error

        # 更新隐层与输出层间连接的权重W[j,k]
        for j in range(self.hnn):
            for k in range(10):
                change = deltas_o[k] * output_h[j] * self.alpha
                self.update_weight(1, j, k, change)

        # 更新隐层与输出层之间连接的权重W[i,j]
        for i in range(64):
            for j in range(self.hnn):
                change = deltas_h[j] * input[i] * self.alpha
                self.update_weight(0, i, j, change)

        return output_o
