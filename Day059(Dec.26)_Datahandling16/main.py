from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from optimizer import *
from util import *
from multi_layer_net import MultiLayerNet


# 0. MNIST 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# print(x_train[0], t_train[0])

train_size = x_train.shape[0]
# print(train_size)   # (60000, 784)
batch_size = 128
max_iterations = 2000

# 1. 실험용 설정
optimizers = {}
optimizers['SGD'] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size = 784, hidden_size_list=[100,100,100,100],
        output_size =10
    )
    train_loss[key] = []

# 2. train ...
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch, t_batch)

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100==0:
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key+":"+str(loss))

# 3. 그래프 그리기
markers = {"SGD" : "o",
          "Momentum" : "x",
          "AdaGrad" : "s",
          "Adam" : "D"
        }
x = np.arange(max_iterations)
plt.figure(figsize=(20,8))
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker = markers[key], markervery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0,1)
plt.legend()
plt.show()