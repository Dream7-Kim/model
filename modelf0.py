import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
# use this is to change the dir to python dir not the IDE terminal dir
i = complex(
    0,
    1,
)


def read(string):
    lines = open(string).readlines()
    row = 5000
    lists = []
    num =0
    for line in lines:
        num += 1
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
        if num == row*4:
            break
    # 将列表变成矩阵，然后重新定义矩阵的形式
    array = onp.array(lists).reshape(row, 4)
    return array

def BW(mass, width, Pb, Pc):
    Pbc = Pb + Pc
    _Pbc = Pbc * onp.array([-1, -1, -1, 1])
    # 需要乘上度规
    Sbc = onp.einsum('ij->i', Pbc * _Pbc)
    # 矩阵相乘的结果缩并就是每一行内积的结果的数组
    # print(Sbc)
    return 1 / (mass**2 - Sbc - i * mass * width)


def phase(theta, rho):
    return rho * np.exp(theta*i)



# def modelf0(var):
#     # up_phif001 = phif001.T * BW(var[1], var[2], Kp, Km) * BW(
#     #     var[3], var[4], Pip, Pim) * phase(var[5], var[7])
#     up_phif001 = phif001.T * BW(1.028, 0, Kp, Km) * BW(
#         1.8, 0.2, Pip, Pim)
#     up = up_phif001
#     # up_1 = onp.vstack([up[0, :], up[1, :]])
#     up_1 = onp.vstack([up[2, :], up[3, :]])
#     conj_up_1 = onp.conj(up_1)
#     up_2 = onp.real(onp.sum(up_1*conj_up_1, axis=0))/2

#     ######################################################

#     low_phif001 = phif001MC.T * BW(1.028, 0, KpMC, KmMC) * BW(
#         1.8, 0.2, PipMC, PimMC)
#     low = low_phif001
#     low_1 = onp.vstack([low[0, :], low[1, :]])
#     conj_low_1 = onp.conj(low_1)
#     low_2 = onp.real(onp.sum(low_1*conj_low_1, axis=0))/2
#     low_3 = onp.average(low_2)
#     result = up_2 / low_3
#     return result

phif001 = read('phif001MC.txt')
phif021 = read('phif021MC.txt')
Kp = read('KpMC.txt')
Km = read('KmMC.txt')
Pip = read('PipMC.txt')
Pim = read('PimMC.txt')

phif001MC = read('phif001MC.txt')
phif021MC = read('phif021MC.txt')
KpMC = read('KpMC.txt')
KmMC = read('KmMC.txt')
PipMC = read('PipMC.txt')
PimMC = read('PimMC.txt')


def modelf0(var):
    up_phif001 = phif001.T * BW(var[0], var[1], Kp, Km) * BW(
        var[2], var[3], Pip, Pim) * var[4]
    up_phif021 = phif021.T * BW(var[0], var[1], Kp, Km) * BW(
        var[2], var[3], Pip, Pim) 
    up = (up_phif001 + up_phif021) * phase(var[5],var[6])
    up_1 = onp.vstack([up[0, :], up[1, :]])
    conj_up_1 = onp.conj(up_1)
    up_2 = onp.real(onp.sum(up_1*conj_up_1, axis=0))/2

    ######################################################

    low_phif001 = phif001MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(
        var[2], var[3], PipMC, PimMC) * var[4]
    low_phif021 = phif021MC.T * BW(var[0], var[1], KpMC, KmMC) * BW(
        var[2], var[3], PipMC, PimMC) 
    low = (low_phif001 + low_phif021) * phase(var[5],var[6])
    low_1 = onp.vstack([low[0, :], low[1, :]])
    conj_low_1 = onp.conj(low_1)
    low_2 = onp.real(onp.sum(low_1*conj_low_1, axis=0))/2
    low_3 = onp.average(low_2)
    result = up_2 / low_3
    return result




# var = onp.array([1.02, 0.00461, 1.37, 0.35, 1, 1, 1, 1])

# test = onp.zeros(5000)
# for n in range(5000):
#     test[n] = 1

# Y = modelf0(var)
# print(".......................\n")
# print(Y.shape)


def readf0(string):
    lines = open(string).readlines()
    lists = []
    num = 0
    for line in lines:
        str = line
        tmp = float(str)
        lists.append(tmp)
        num += 1
        if num == 5000:
            break
    return onp.array(lists)




