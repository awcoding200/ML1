import numpy

import numpy,matplotlib
from matplotlib import pyplot as plt

class Abalone:

    # Instantiate the Abalone dataset
    #
    # input: None
    # output: None
    #
    def __init__(self):
        X = numpy.genfromtxt('s3/abalone.csv',delimiter=',')
        self.N = X[X[:,0] <2][:,1:-1]
        self.I = X[X[:,0]==2][:,1:-1]

    # Plot a histogram of the projected data
    #
    # input: the projection vector and the name of the projection
    # output: None
    #
    def plot(self,w,name):
        plt.figure(figsize=(6,2))
        plt.xlabel(name)
        plt.ylabel('# examples')
        plt.hist(numpy.dot(self.I,w),bins=25, alpha=0.8,color='red',label='infant')
        plt.hist(numpy.dot(self.N,w),bins=25, alpha=0.8,color='gray',label='noninfant')
        plt.legend()
        plt.show()


# def fisher(a, b):
    # mu_a, mu_b = a.mean(axis=0).reshape(-1,1), b.mean(axis=0).reshape(-1,1)
    # Sw = numpy.cov(a.T) + numpy.cov(b.T)
    # inv_S = numpy.linalg.inv(Sw)
    # res = inv_S.dot(mu_a-mu_b)  # the trick
    # return res

def fisher(X1,X2):
    ##### Replace by your code
    # calculate m1,m2
    # 计算每个类别的均值向量，并将其转换为列向量
    m1 = numpy.mean(X1, axis=0).reshape(-1, 1)  # 形状为 (n_features, 1)
    m2 = numpy.mean(X2, axis=0).reshape(-1, 1)  # 形状为 (n_features, 1)
    # 计算类间散布矩阵 Sb
    mean_diff = m1 - m2  # 形状为 (n_features, 1)

    # 计算每个类别的协方差矩阵
    S1 = numpy.cov(X1, rowvar=False)
    S2 = numpy.cov(X2, rowvar=False)  

    # 计算类内散布矩阵 Sw
    Sw = S1 + S2  # 形状为 (n_features, n_features)

    # 计算 Sw 的逆矩阵，如果 Sw 是奇异的，可以使用伪逆
    try:
        Sw_inv = numpy.linalg.inv(Sw)
    except numpy.linalg.LinAlgError:
        Sw_inv = numpy.linalg.pinv(Sw)

    # 计算 Fisher 判别向量 w = Sw^{-1} (m1 - m2)
    w = Sw_inv @ mean_diff  # 形状为 (n_features, 1)

    return w
    #####
    
def objective(X1,X2,w):
    ##### Replace by your code
    # Compute the mean vectors
    # 确保 w 是二维列向量
    w = numpy.asarray(w).reshape(-1, 1)  # 形状为 (n_features, 1)
    
    # 计算每个类别的均值向量，并转换为列向量
    m1 = numpy.mean(X1, axis=0).reshape(-1, 1)  # 形状为 (n_features, 1)
    m2 = numpy.mean(X2, axis=0).reshape(-1, 1)  # 形状为 (n_features, 1)
    
    # 计算 Sb = (m1 - m2) (m1 - m2)^T
    mean_diff = m1 - m2  # 形状为 (n_features, 1)
    Sb = numpy.matmul(mean_diff, mean_diff.T)  # 形状为 (n_features, n_features)
    
    # 计算类内散布矩阵 S_w
    S1 = numpy.cov(X1, rowvar=False)  # 形状为 (n_features, n_features)
    S2 = numpy.cov(X2, rowvar=False)  # 形状为 (n_features, n_features)
    S_w = S1 + S2  # 形状为 (n_features, n_features)
    
    # 计算分子: w^T Sb w
    temp1 = numpy.matmul(w.T, Sb)        # 形状为 (1, n_features)
    numerator = numpy.matmul(temp1, w)[0][0]  # 标量
    
    # 计算分母: w^T S_w w
    temp2 = numpy.matmul(w.T, S_w)       # 形状为 (1, n_features)
    denominator = numpy.matmul(temp2, w)[0][0]  # 标量
    
    # 防止分母为零
    if denominator == 0:
        return numpy.inf
    
    J = numerator / denominator
    return J
    #####
    
def expand(X):
    ##### Replace by your code
    n_samples, d = X.shape
    num_quad = d * (d + 1) // 2
    Z = numpy.zeros((n_samples, d + num_quad))
    Z[:, :d] = X
    idx = d
    for i in range(d):
        for j in range(i, d):
            Z[:, idx] = X[:, i] * X[:, j]
            idx += 1
    return Z
    #####



if __name__ == '__main__':
    # Load the data
    abalone = Abalone()

    # Print dataset size for each class
    print(abalone.I.shape, abalone.N.shape)



    # Compute the first feature
    w = numpy.array([1,0,0,0,0,0,0])
    print(objective(abalone.I, abalone.N, w))
    print('Fisher discriminant vector shape:', w.shape)
    # print('Fisher discriminant vector:', w)


    # Mean
    m1 = numpy.mean(abalone.I, axis=0)
    m2 = numpy.mean(abalone.N, axis=0)
    w = m1-m2
    print('Means shape:', w.shape)
    print('Means objective:', objective(abalone.I, abalone.N, w))
    # abalone.plot(w,'Means')


    w = fisher(abalone.I, abalone.N)
    print('Fisher discriminant vector shape:', w.shape)
    # print('Fisher discriminant vector:', w)
    print('Fisher objective:', objective(abalone.I, abalone.N, w))
    # abalone.plot(w,'Fisher')

