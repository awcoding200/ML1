import numpy
import matplotlib.pyplot as plt
from solutions import fisher

def fisher(X1, X2):
    mu_a, mu_b = X1.mean(axis=0).reshape(-1,1), X2.mean(axis=0).reshape(-1,1)
    Sw = numpy.cov(X1.T) + numpy.cov(X2.T)
    inv_S = numpy.linalg.inv(Sw)
    res = inv_S.dot(mu_a-mu_b)  # the trick
    return res

# a = numpy.random.multivariate_normal((1.5, 3), [[0.5, 0], [0, .05]], 30)
# b = numpy.random.multivariate_normal((4, 1.5), [[0.5, 0], [0, .05]], 30)
# plt.plot(a[:,0], a[:,1], 'b.', b[:,0], b[:,1], 'r.')
# # mu_a, mu_b = a.mean(axis=0).reshape(-1,1), b.mean(axis=0).reshape(-1,1)
# # Sw = numpy.cov(a.T) + numpy.cov(b.T)
# # inv_S = numpy.linalg.inv(Sw)
# # res = inv_S.dot(mu_a-mu_b)  # the trick

# res = fisher(a, b)

# ####
# # more general solution
# #
# # Sb = (mu_a-mu_b)*((mu_a-mu_b).T)
# # eig_vals, eig_vecs = numpy.linalg.eig(inv_S.dot(Sb))
# # res = sorted(zip(eig_vals, eig_vecs), reverse=True)[0][1] # take only eigenvec corresponding to largest (and the only one) eigenvalue
# # res = res / numpy.linalg.norm(res)

# plt.plot([-res[0], res[0]], [-res[1], res[1]]) # this is the solution
# # plt.plot(mu_a[0], mu_a[1], 'cx')
# # plt.plot(mu_b[0], mu_b[1], 'yx')
# plt.gca().axis('square')

# # let's project data point on it
# r = res.reshape(2,)
# n2 = numpy.linalg.norm(r)**2
# for pt in a:
#     prj = r * r.dot(pt) / n2
#     plt.plot([prj[0], pt[0]], [prj[1], pt[1]], 'b.:', alpha=0.2)
# for pt in b:
#     prj = r * r.dot(pt) / n2
#     plt.plot([prj[0], pt[0]], [prj[1], pt[1]], 'r.:', alpha=0.2)

# plt.show()





m1 = numpy.array([-1,-1]).reshape(-1,1)
m2 = numpy.array([1,1]).reshape(-1,1)
# 计算类间散布矩阵 Sb
mean_diff = m1 - m2  # 形状为 (n_features, 1)
print(mean_diff)
# 计算每个类别的协方差矩阵
S1 = numpy.array([[2,0],[0,1]])
S2 = numpy.array([[2,0],[0,1]])
print(S1)
print(f"S2 = {S2}")

Sb = numpy.matmul(mean_diff, mean_diff.T)  # 形状为 (n_features, n_features)
print(f'Sb = {Sb}')

# 计算类内散布矩阵 Sw
Sw = S1 + S2  # 形状为 (n_features, n_features)
print(f'Sw = {Sw}')

# 计算 Sw 的逆矩阵，如果 Sw 是奇异的，可以使用伪逆
try:
    Sw_inv = numpy.linalg.inv(Sw)
except numpy.linalg.LinAlgError:
    Sw_inv = numpy.linalg.pinv(Sw)

print(f'Sw_inv = {Sw_inv}')

# 计算 Fisher 判别向量 w = Sw^{-1} (m1 - m2)
w = Sw_inv @ mean_diff  # 形状为 (n_features, 1)

print(w)


# get orthogonal vector to w
w = w / numpy.linalg.norm(w)
orth = numpy.array([[0, -1], [1, 0]]) @ w
print(f'orth =\n{orth}')


# if u have a vec x = [x1, x2]
#  after expasion x = [x1, x2, x1^2, x2^2, x1*x2]