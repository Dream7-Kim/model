import numpy as onp
import jax.numpy as np
import os
import jax
import time

i = complex(
    0,
    1,
)

def read(string):
    lines = open(string).readlines()
    row = int(len(lines) / 4)
    lists = []
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
    # 将列表变成矩阵，然后重新定义矩阵的形式
    array = onp.array(lists).reshape(row, 4)
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


phif001 = read('phif001.txt')
phif021 = read('phif021.txt')
phif201 = read('phif201.txt')
Kp = read('Kp.txt')
Km = read('Km.txt')
Pip = read('Pip.txt')
Pim = read('Pim.txt')

phif001MC = read('phif001MC.txt')
phif021MC = read('phif021MC.txt')
phif201MC = read('phif201MC.txt')
KpMC = read('KpMC.txt')
KmMC = read('KmMC.txt')
PipMC = read('PipMC.txt')
PimMC = read('PimMC.txt')

num = 1

def model(var):    
    up = np.zeros((4, 5254))
    # 添加共振态 数据
    for i in range(num):
        up_phif001 = phif001.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[2 + 12*i], var[3 + 12*i], Pip, Pim) * phase(var[6 + 12*i] - var[8 + 12*i], var[7 + 12*i])
        up_phif021 = phif021.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[2 + 12*i], var[3 + 12*i], Pip, Pim)
        up_phif201 = phif201.T * BW(var[0 + 12*i], var[1 + 12*i], Kp, Km) * BW(var[4 + 12*i], var[5 + 12*i], Pip, Pim) * phase(var[10 + 12*i], var[11 + 12*i])
        
        up = up + (up_phif001 + up_phif021) * phase(var[8 + 12*i], var[9 + 12*i]) + up_phif201
    # print(up.shape)

    up_1 = np.vstack([up[0, :], up[1, :]])
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1*conj_up_1, axis=0))/2 # (5254, )
    

    # 添加共振态 蒙卡
    for i in range(num):
        low_phif001 = phif001MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[2 + 12*i], var[3 + 12*i], PipMC, PimMC) * phase(var[6 + 12*i] - var[8 + 12*i], var[7 + 12*i])
        low_phif021 = phif021MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[2 + 12*i], var[3 + 12*i], PipMC, PimMC)
        low_phif201 = phif201MC.T * BW(var[0 + 12*i], var[1 + 12*i], KpMC, KmMC) * BW(var[4 + 12*i], var[5 + 12*i], PipMC, PimMC) * phase(var[10 + 12*i], var[11 + 12*i])
    
        low = (low_phif001 + low_phif021) * phase(var[8 + 12*i], var[9 + 12*i]) + low_phif201

    low_1 = np.vstack([low[0, :], low[1, :]])
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1*conj_low_1, axis=0))
    # print(low_2.shape)
    low_3 = np.average(low_2) # scalar
    # return np.power(low_3, 5254)
    

    # result = up_2_func(var) / low_3_func(var)
    # result = np.log(up_2) - 5254 * np.log(low_3)
    # result = np.sum(result)
    # result1 = np.prod(result)

    result1 = np.sum(np.log(up_2))
    result1 = result1 - 5254 * np.log(low_3)

    return result1




np.set_printoptions(precision=16,threshold=np.inf)



# model([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

var_ = onp.zeros(12 * num)
for i in range(12 * num):
    var_[i] = 0.1

grad = jax.grad(model)
start = time.time()
print(grad(var_))
end = time.time()
print('Execution time: ', float(end-start))

start = time.time()
print(grad(var_))
end = time.time()
print('Execution time: ', float(end-start))

start = time.time()
print(grad(var_))
end = time.time()
print('Execution time: ', float(end-start))