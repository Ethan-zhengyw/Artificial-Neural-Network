#-*- coding: utf-8 -*-
from pprint import pprint
import random
from math import exp


class NetTable:
    
    def __init__(self, nh_ = 80, alpha_ = 0.05, beta_ = 0.2):
        """ 初始化权重

        :param nh_: 设置隐层节点的数目
        :param alpha_: 设置改变步长
        :param beta_: 设置学习速率

        """
        # input, hidden, output layer's node number
        self.ni = 64
        self.nh = nh_
        self.no = 10

        self.alpha = alpha_
        self.beta = beta_

        # 两个张储存 input-hidden 和 hidden-output 之间权重的表
        self.w_ih = self.makeMatrix(self.ni, self.nh, True)
        self.w_ho = self.makeMatrix(self.nh, self.no, True)

        # 前一个∆W，初始化为0
        self.c_ih = self.makeMatrix(self.ni, self.nh, False)
        self.c_ho = self.makeMatrix(self.nh, self.no, False)


    def train(self, records):
        for record in records:
            self.back_propagate(record[:64], record[64])


    def feed_forward(self, input):
        """ 信息正向传递,前馈查询

        :param input: 一个含64个数字的列表，代表一个数字的64维的特征向量
        :returns:
            `output_o` - 对应每一个隐藏层的输出
            `output_h` - 对应0~9的10个数字的可能性

        """

        # 每一层各个节点的输出, output_i已知
        output_i = input
        output_h = [1.0] * self.nh
        output_o = [1.0] * self.no

        # i, j, k分别是输入层，隐层，输出层的Node的下标

        # 计算隐藏层的输出结果
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += (output_i[i] * self.w_ih[i][j])
            output_h[j] = self.sigmoid(sum)

        # 计算输出层的输出结果
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (output_h[j] * self.w_ho[j][k])
            output_o[k] = self.sigmoid(sum)

        return output_h, output_o

    
    def back_propagate(self, input, target):
        """ 误差后向传播调整权重

        :param input: 64维特征向量, 每一个值的范围为 0~16
        :param target: 一个与input相对应的整数 0~9

        """
        # targets中与正确数字对应的，值设为0.9，其余的设为0.1
        targets = [0.1] * self.no
        targets[target] = 1

        # 获取将input推入网络后，每个节点的输出
        output_i = input
        output_h, output_o = self.feed_forward(input)

        # 计算输出层每一个节点处的误差
        errors_o = [0.0] * self.no
        for k in range(self.no):
            errors_o[k] = self.Errors(output_o[k], targets[k])

        
        # 更新隐层与输出层间连接的权重W[j,k]
        for j in range(self.nh):
            for k in range(self.no):
                change = self.beta * errors_o[k] * output_h[j]
                self.update_weight(1, j, k, change)

        ## 计算隐藏的每一个节点的误差
        #errors_h = [0.0] * self.nh
        #for j in range(self.nh):
        #    for k in range(self.no):
        #        errors_h[j] += (output_o[k] * self.w_ho[j][k])
        #    errors_h[j] *= (output_h[j] * (1 - output_h[j]))

        ## 更新输入层与隐藏层之间的权重W[i,j]
        #for i in range(self.ni):
        #    for j in range(self.nh):
        #        change = self.beta * errors_h[j] * output_i[i]
        #        self.update_weight(0, i, j, change)


    def update_weight(self, layer, fromid, toid, change):

        (target_w, target_c) = (self.w_ih, self.c_ih) \
                if layer == 0 else (self.w_ho, self.c_ho)

        target_w[fromid][toid] += \
                (change + self.alpha * target_c[fromid][toid])

        target_c[fromid][toid] = change + self.alpha * target_c[fromid][toid]


    def get_weight(self, layer, fromid, toid):
        """ 查询两个节点之间的权重

        :param layer: 指定查询哪一层之间的节点，0代表输入层与隐层；1代表隐层与输出层
        :param fromid: 
            当`layer=0`时: 0 < fromid < 64, 0 < toid < hiddenNodeNum
            当`layer=1`时: 0 < fromid < hiddenNodeNum, 0 < toid < 10

        """
        target = self.ih if layer == 0 else self.ho
        weight = target[fromid][toid]

        return weight


    def makeMatrix(self, row, col, rand):
        """ 建立一个二维矩阵

        :param random: 是否随机赋值,False的话就全部设置为0

        """
        matrix = []
        for i in range(row):
            matrix.append([])
            for j in range(col):
                matrix[i].append(random.uniform(-0.2, 0.2)\
                        if rand else 0)
        return matrix


    #输出层的Error函数
    def Errors(self, output, expect):
        return output * (1 - output) * (expect - output)


    def sigmoid(self, n):
        return 1.0 / (1.0 + exp(-n))
