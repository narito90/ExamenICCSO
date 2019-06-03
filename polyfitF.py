import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def polyfit(x, y, n):
    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen, n + 1), dtype = int)
    xT = np.matrix(x)
    yT = np.matrix(y)
    c2 = np.power(xT, 2)
    c1 = one[:, [1]]
    A = np.hstack([c1, xT, c2])

    def inv(A):
        return np.linalg.inv(A)

    def trans(A):
        return A.getT()

    def prod(A,B):
        return np.dot(A,B)

    AtA = inv(prod(trans(A), A))
    print(AtA)

    up = prod(AtA, trans(A))

    u = prod(up, trans(yT))
    print(u)

#x = [1, -2, 3, 4]
#y = [1, 4, 9, 16]

df1 = pd.read_csv('exam_B_dataset.csv')
X = df1.iloc[:, :-1].values
y = df1.iloc[:, 1].values
plt.plot(polyfit(X, y, 1))
plt.show()
