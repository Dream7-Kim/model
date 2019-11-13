import numpy as onp
import jax.numpy as np
import os

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


def model(phif001, phif021, phif201, Kp, Km, Pip, Pim, phif001MC, phif021MC,
          phif201MC, KpMC, KmMC, PipMC, PimMC, massphi, widthphi, massf0,
          widthf0, massf2, widthf2, theta_phif0_1, rho_phif0_1, theta_phif0_2,
          rho_phif0_2, theta_phif2_1, rho_phif2_1):
    # 添加共振态 数据
    up_phif001 = phif001.T * BW(massphi, widthphi, Kp, Km) * BW(
        massf0, widthf0, Pip, Pim) * phase(theta_phif0_1 - theta_phif0_2,
                                           rho_phif0_1)
    up_phif021 = phif021.T * BW(massphi, widthphi, Kp, Km) * BW(
        massf0, widthf0, Pip, Pim)
    up_phif201 = phif201.T * BW(massphi, widthphi, Kp, Km) * BW(
        massf2, widthf2, Pip, Pim) * phase(theta_phif2_1, rho_phif2_1)
    up = (up_phif001 + up_phif021) * \
        phase(theta_phif0_2, rho_phif0_2) + up_phif201
    up_1 = np.vstack([up[0, :], up[1, :]])
    conj_up_1 = np.conj(up_1)
    up_2 = np.real(np.sum(up_1*conj_up_1, axis=0))/2
    # 添加共振态 蒙卡
    low_phif001 = phif001MC.T * BW(massphi, widthphi, KpMC, KmMC) * BW(
        massf0, widthf0, PipMC, PimMC) * phase(theta_phif0_1 - theta_phif0_2,
                                               rho_phif0_1)
    low_phif021 = phif021MC.T * BW(massphi, widthphi, KpMC, KmMC) * BW(
        massf0, widthf0, PipMC, PimMC)
    low_phif201 = phif201MC.T * BW(massphi, widthphi, KpMC, KmMC) * BW(
        massf2, widthf2, PipMC, PimMC) * phase(theta_phif2_1, rho_phif2_1)
    low = (low_phif001 + low_phif021) * \
        phase(theta_phif0_2, rho_phif0_2) + low_phif201
    # print(low)
    low_1 = np.vstack([low[0, :], low[1, :]])
    conj_low_1 = np.conj(low_1)
    low_2 = np.real(np.sum(low_1*conj_low_1, axis=0))
    # print(low_2)
    low_3 = np.average(low_2)
    # print(low_3)
    result = up_2 / low_3
    print(result)
    result1 = np.prod(result)
    # print(result1)
    return np.log(result1)


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
#print(BW(1, 1, Kp, Km))
# print(
np.set_printoptions(precision=16,threshold=np.inf)
print(model(phif001, phif021, phif201, Kp, Km, Pip, Pim, phif001MC, phif021MC,
            phif201MC, KpMC, KmMC, PipMC, PimMC, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

#lines = txt.readlines()
# lines.count('\n')
