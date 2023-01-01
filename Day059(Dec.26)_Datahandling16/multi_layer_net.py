# multi_layer_net.py
import numpy as np
from collections import OrderedDict
from layer import *
from gradient import *


class MultiLayerNet:
    """
    input_size -> 입력 크기 784 -> MNIST
    hidden_size_list -> 각 은닉층의 뉴런 수를 담은 리스트 -> [100,100,100]
    output_size -> 출력 크기 (MNIST -> 10)
    activation -> relu or sigmoid
    weight_init_std -> 가중치 표준편차 (0.01)
        'relu -> he -> He 초기값
        'sigmoid -> xavier로 지정하면 Xavier 초기값으로 설정
    weight_decay_lambda -> 가중치 감소(L2 법칙)의 세기
    """
    def __init__(self, input_size, hidden_size_list, output_size, 
                activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_number = len(hidden_size_list)
        self.weight_decay_lambda=weight_decay_lambda
        self.params ={}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 계층생성
        activation_layer = {"sigmoid":Sigmoid,
                            "relu" : Relu
                            }
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_number +1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' +str(idx)],
                                                      self.params['b' +str(idx)])
            self.layers['Activation_function'+str(idx)] = activation_layer[activation]()

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        # 가중치 초기화
        all_size_list = [self.input_size]+\
            self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0/all_size_list[idx-1])   # Relu
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0/all_size_list[idx-1])   # Sigmoid
            
            self.params["W"+str(idx)] = scale*\
                np.random(all_size_list[idx-1], all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        
        for idx in range(1, self.hidden_layer_number+2):
            W = self.params["w"+str(idx)]
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(W**2)

        return self.last_layer.forward(y,t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y==t)/float(x.shape[0])

        return acc
    
    def numerical_gradient(self, x, t):
        """
        grads['W1'], grads['W2'], grads['W3'].... 각 층의 가중치
        grads['b1'], grads['b2'], grads['b3'].... 각 층의 편향
        """
        def loss_W(W): return self.loss(x,t)

        grads = {}
        for idx in range(1, self.hidden_layer_number+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W'+str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b'+str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.hidden_layer_number + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW\
                 + self.weight_decay_lambda*self.layers['Affine'+str(idx)].W
            grads['b' + str(idx)] = self.last_layer['Affine' + str(idx)].db

        return grads           