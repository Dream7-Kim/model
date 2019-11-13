#%%
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as onp
import jax.numpy as np
import os
import jax
import time
import scipy.optimize as opt


# use this is to change the dir to python dir not the IDE terminal dir
# os.chdir('/home/sean/JAX')
# this is to define the complex base
i = complex(
    0,
    1,
)


def read(string):
    lines = open(string).readlines()
    row = int(len(lines) / 4)
    lists = []
    num = 0
    for line in lines:
        str = line.replace('[', '')
        str = str.replace(']', '')
        str = str.strip()
        tmp = float(str)
        lists.append(tmp)
        num += 1
        if num == 5000 * 4:
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


def modelf0(var):
    phif001 = read('phif001MC.txt')
    phif021 = read('phif021MC.txt')
    phif201 = read('phif201MC.txt')
    Kp = read('KpMC.txt')
    Km = read('KmMC.txt')
    Pip = read('PipMC.txt')
    Pim = read('PimMC.txt')

    phif001MC = read('phif001MC.txt')
    phif021MC = read('phif021MC.txt')
    phif201MC = read('phif201MC.txt')
    KpMC = read('KpMC.txt')
    KmMC = read('KmMC.txt')
    PipMC = read('PipMC.txt')
    PimMC = read('PimMC.txt')
    # up_phif001 = phif001.T * BW(massphi, widthphi, Kp, Km) * BW(
    #     massf0, widthf0, Pip, Pim) * phase(theta_phif0_1 - theta_phif0_2,
    #                                        rho_phif0_1)
    # up_phif021 = phif021.T * BW(massphi, widthphi, Kp, Km) * BW(
    #     massf0, widthf0, Pip, Pim)
    # up = (up_phif001 + up_phif021) * phase(theta_phif0_2, rho_phif0_2)
    ####
    up_phif001 = phif001.T * BW(var[1], var[2], Kp, Km) * BW(
        var[3], var[4], Pip, Pim) * phase(var[5] - var[6],
                                          var[7])
    up_phif021 = phif021.T * BW(var[1], var[2], Kp, Km) * BW(
        var[3], var[4], Pip, Pim)
    up = (up_phif001 + up_phif021) * phase(var[6], var[8])
    up_1 = np.vstack([up[0, :], up[1, :]])
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1*conj_up_1, axis=0))/2
    ####
    low_phif001 = phif001MC.T * BW(var[1], var[2], KpMC, KmMC) * BW(
        var[3], var[4], PipMC, PimMC) * phase(var[5] - var[6],
                                              var[7])
    low_phif021 = phif021MC.T * BW(var[1], var[2], KpMC, KmMC) * BW(
        var[3], var[4], PipMC, PimMC)
    low = (low_phif001 + low_phif021) * phase(var[6], var[8])
    low_1 = np.vstack([low[0, :], low[1, :]])
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1*conj_low_1, axis=0))/2
    low_3 = np.average(low_2)
    result = up_2 / low_3
    return result


var_weight = np.array([0.0, 1019.461, 4.249, 990., 55., 1., 1., 1., 1.])
x = modelf0(var_weight)
Pip = read('PipMC.txt')
Pim = read('PimMC.txt')
y = Pip + Pim 
print(y.shape, x.shape)

plt.plot(x,y)

plt.savefig('model_f0.png')
# %%
