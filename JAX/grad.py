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




def model(var):    
    # result = up_2_func(var) / low_3_func(var)
    result = np.log(up_2_func(var)) - 5254 * np.log(low_3_func(var))
    # result1 = np.prod(result)
    return result


def up_2_func(var):
    # 添加共振态 数据
    up_phif001 = phif001.T * BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim) * phase(var[6] - var[8], var[7])
    up_phif021 = phif021.T * BW(var[0], var[1], Kp, Km) * BW(var[2], var[3], Pip, Pim)
    up_phif201 = phif201.T * BW(var[0], var[1], Kp, Km) * BW(var[4], var[5], Pip, Pim) * phase(var[10], var[11])
    up = (up_phif001 + up_phif021) * phase(var[8], var[9]) + up_phif201
    up_1 = np.vstack([up[0, :], up[1, :]])
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1*conj_up_1, axis=0))/2 # (5254, )
    # print(up_2.shape)

    up_3 = np.prod(up_2)

    return up_3


def low_3_func(var):
    # 添加共振态 蒙卡
    low_phif001 = phif001MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC) * phase(var[6] - var[8], var[7])
    low_phif021 = phif021MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(var[2], var[3], PipMC, PimMC)
    low_phif201 = phif201MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(var[4], var[5], PipMC, PimMC) * phase(var[10], var[11])
    low = (low_phif001 + low_phif021) * phase(var[8], var[9]) + low_phif201
    low_1 = np.vstack([low[0, :], low[1, :]])
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1*conj_low_1, axis=0))
    # print(low_2.shape)
    low_3 = np.average(low_2) # scalar
    # return np.power(low_3, 5254)
    return low_3


np.set_printoptions(precision=16,threshold=np.inf)


# low_3_grad = jax.grad(low_3_func)
# start = time.time()
# print(low_3_grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
# end = time.time()
# print('Execution time: ', float(end-start))

# start = time.time()
# print(low_3_grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
# end = time.time()
# print('Execution time: ', float(end-start))

# up_2_grad = jax.jit(jax.jacrev(up_2_func))
# start = time.time()
# print(up_2_grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
# end = time.time()
# print('Execution time: ', float(end-start))

# start = time.time()
# print(up_2_grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
# end = time.time()
# print('Execution time: ', float(end-start))



# start = time.time()
# print(model([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
# end = time.time()
# print('Model Cal time: ', float(end - start))



grad = jax.grad(model)
start = time.time()
print(grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
end = time.time()
print('Execution time: ', float(end-start))

start = time.time()
print(grad([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
end = time.time()
print('Execution time: ', float(end-start))