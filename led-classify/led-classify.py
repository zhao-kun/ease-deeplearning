import numpy as np

from train import train, predict
from dense import Dense
from activation import Tanh
from loss import mse, mse_prime


def load_data():
    #     -------- <--0
    #     |      |
    #     | <-1  | <--2 
    #     -------- <--3
    #     |      |
    #     | <-4  | <--5
    #     -------- <--6
    # 
    x0 = [1, 1, 1,0 ,1 ,1, 1] #0
    x1 = [0, 0, 1,0 ,0 ,1, 0] #1
    x2 = [1, 0, 1,1 ,1 ,0, 1] #2
    x3 = [1, 0, 1,1 ,0 ,1, 1] #3
    x4 = [0, 1, 1,1 ,0 ,1, 0] #4
    x5 = [1, 1, 0,1 ,0 ,1, 1] #5
    x6 = [1, 1, 0,1 ,1 ,1, 1] #6
    x7 = [1, 0, 1,0 ,0 ,1, 0] #7
    x8 = [1, 1, 1,1 ,1 ,1, 1] #8
    x9 = [1, 1, 1,1 ,0 ,1, 1] #9
    x_train = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    y_train = [[1 if i == j else 0 for i in range(10)] for j in range(10)]

    x_train = np.array(x_train).reshape(10, 7, 1)
    y_train = np.array(y_train).reshape(10, 10, 1)

def build_network():
    network = [
        Dense(7,64),
        Tanh(),
        Dense(64,10),
    ]
    return network

def test_network(network, x_train):
    for x in x_train:
        output = predict(network, x)
        print(f'{x} => {np.argmax(output)}')
        

def led_classify():
    x_train, y_train = load_data();
    network = build_network()
    train(network, mse, mse_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, interval=100, verbose = True)
    
    test_network(network, x_train)


if __name__ == "__main__":
    led_classify()