from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as onp
import jax.numpy as np
import os
import jax
import time
import scipy.optimize as opt
# import modelf0
import seaborn as sns

i = complex(
    0,
    1,
)



def read(string):
    lines = open(string).readlines()
    lists = []
    num = 0
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
        num = num + 1
        if(num == 5000*4):
            break
    # 将列表变成矩阵，然后重新定义矩阵的形式
    array = onp.array(lists).reshape(int(num/4), 4)
    return array

def BW(mass, width, Pb, Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * np.array([-1, -1, -1, 1])
    # 需要乘上度规
    Sbc = np.einsum('ij->i', Pbc * _Pbc)
    # 矩阵相乘的结果缩并就是每一行内积的结果的数组
    # print(Sbc)
    return 1 / (mass**2 - Sbc - i * mass * width)


def phase(theta, rho):
    return rho * np.exp(theta)

# phif001 = read('phif001.txt')
# phif021 = read('phif021.txt')
# phif201 = read('phif201.txt')
# Kp = read('Kp.txt')
# Km = read('Km.txt')
# Pip = read('Pip.txt')
# Pim = read('Pim.txt')

phif001 = read('phif001MC.txt')
phif021 = read('phif021MC.txt')
# phif201 = read('phif201MC.txt')
Kp = read('KpMC.txt')
Km = read('KmMC.txt')
Pip = read('PipMC.txt')
Pim = read('PimMC.txt')

phif001MC = read('phif001MC.txt')
phif021MC = read('phif021MC.txt')
# phif201MC = read('phif201MC.txt')
KpMC = read('KpMC.txt')
KmMC = read('KmMC.txt')
PipMC = read('PipMC.txt')
PimMC = read('PimMC.txt')


def modelf0(var):
    up_phif001 = phif001.T * BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim) * var[4]
    up_phif021 = phif021.T * BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim) 
    up = (up_phif001 + up_phif021) * phase(var[5],var[6])
    up_1 = onp.vstack([up[0, :], up[1, :]])
    conj_up_1 = onp.conj(up_1)
    up_2 = onp.real(onp.sum(up_1*conj_up_1, axis=0))/2

    ######################################################

    low_phif001 = phif001MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC) * var[4]
    low_phif021 = phif021MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC) 
    low = (low_phif001 + low_phif021) * phase(var[5],var[6])
    low_1 = onp.vstack([low[0, :], low[1, :]])
    conj_low_1 = onp.conj(low_1)
    low_2 = onp.real(onp.sum(low_1*conj_low_1, axis=0))/2
    low_3 = onp.average(low_2)

    result = up_2 / low_3
    return result


var_weight = np.array([1.019, 0.00461, 1.37, 0.35, 1., 1., 1.])
weight = modelf0(var_weight)

# print(weight)
# plt.hist(weight, 100, density=True, facecolor='g', alpha=0.75)
sns.distplot(weight)
plt.savefig('histogram.png')

num = 1

def model(var):    
    up = np.zeros((4, 5000))
    # 添加共振态 数据
    for i in range(num):
        up_phif001 = phif001.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[2 + 12*i], var[3 + 12*i], Pip, Pim) * var[4+12*i]
        up_phif021 = phif021.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[2 + 12*i], var[3 + 12*i], Pip, Pim)
        # up_phif201 = phif201.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[4 + 12*i], var[5 + 12*i], Pip, Pim) * phase(var[10 + 12*i], var[11 + 12*i])
        
        # up = up + (up_phif001 + up_phif021) * phase(var[8 + 12*i], var[9 + 12*i]) + up_phif201
        up = up + (up_phif001 + up_phif021) * phase(var[5 + 12*i], var[6 + 12*i])
    # print(up.shape)

    up_1 = np.vstack([up[0, :], up[1, :]])
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1*conj_up_1, axis=0))/2 * weight  # (5254, )
    
    low = np.zeros((4, 5000))
    # 添加共振态 蒙卡
    for i in range(num): 
        low_phif001 = phif001MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[2 + 12*i], var[3 + 12*i], PipMC, PimMC) * var[4 + 12*i]
        low_phif021 = phif021MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[2 + 12*i], var[3 + 12*i], PipMC, PimMC)
        # low_phif201 = phif201MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[4 + 12*i], var[5 + 12*i], PipMC, PimMC) * phase(var[10 + 12*i], var[11 + 12*i])
    
        # low = (low_phif001 + low_phif021) * phase(var[8 + 12*i], var[9 + 12*i]) + low_phif201
        low = low + (low_phif001 + low_phif021) * phase(var[5 + 12*i], var[6 + 12*i])

    low_1 = np.vstack([low[0, :], low[1, :]])
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1*conj_low_1, axis=0))/2 
    # print(low_2.shape)
    low_3 = np.average(low_2) # scalar
    # return np.power(low_3, 5254)
    

    # result = up_2_func(var) / low_3_func(var)
    # result = np.log(up_2) - 5254 * np.log(low_3)
    # result = np.sum(result)
    # result1 = np.prod(result)

    result1 = np.sum(np.log(up_2))
    result1 = result1 - 5000 * np.log(low_3)

    return -result1




# np.set_printoptions(precision=16,threshold=np.inf)



# model([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# var_ = onp.zeros(12 * num)
# for i in range(12 * num):
#     var_[i] = 1 * i
var_ = [1. , 0.004, 1.0, 0.05, 1., 1., 1.]

######################### scipy optimize
grad = jax.jit(jax.grad(model))

print('Initial value: ', model(var_))
print('Initial Grad: ', grad(var_))

start = time.time()

res = opt.minimize(model, var_, method='BFGS', jac=grad, options={'disp':True})

end = time.time()
print('\n\nExec time: ', float(end - start), '\n\n')

print('inverse of hessian matrix \n', res.hess_inv)
# res = opt.minimize(model, var_, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print('Result: ', res.x)
print('Model in point: ', model(res.x))


# # ######################### Draw the graph
# x = onp.linspace(-1, 1, 100)
# y = onp.linspace(-1, 1, 100)
# X, Y = onp.meshgrid(x, y)
# # print(X.shape)
# # print(Y.shape)
# Z = onp.zeros((100, 100))
# xidx = 0
# yidx = 0
# for xi in x:
#     for yi in y:
#         Z[xidx, yidx] = model([1., 1., xi, yi,  1., 1., 1., 1., 1., 1., 1., 1.])
#         # print(yidx)
#         yidx = yidx + 1
#     xidx = xidx + 1
#     yidx = 0
#     print(xidx)
# # print(xidx, yidx)


# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.savefig('model.png')

# print(model([1., 1., 1., 1.,  1., 1., 1., 1., 1., 1., 1., 1.]))



