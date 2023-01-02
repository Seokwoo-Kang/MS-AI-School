# b값이 변화에 따른 경사도 변화
# b값에 따라서 그래프가 어떻게 변하는지 확인 !!

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(t):
    return 1/(1+np.exp(-t))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')    # b 0.5인 경우
plt.plot(x, y2, 'g')                    # b 1인 경우
plt.plot(x, y3, 'b', linestyle='--')    # b 1.5인 경우
plt.plot([0,0], [1.0, 0.0], ':')        # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()