# %%
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
import sys


def fun(var):
    return np.sin(var[0]) * np.cos(var[1])


grad = jax.jit(jax.grad(fun))

fun_and_grad = jax.jit(jax.value_and_grad(fun))

def global_fun(var):
    return fun_and_grad(var)[0], fun_and_grad(var)[1]

# ######################### Draw the graph
# x = onp.linspace(-10, 10, 1000)
# y = onp.linspace(-10, 10, 1000)
# X, Y = onp.meshgrid(x, y)
# # print(X.shape)
# # print(Y.shape)
# Z = onp.zeros((1000, 1000))
# xidx = 0
# yidx = 0
# for xi in x:
#     for yi in y:
#         Z[xidx, yidx] = fun([xi, yi])
#         # print(yidx)
#         yidx = yidx + 1
#     xidx = xidx + 1
#     yidx = 0
#     print(xidx)


# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.savefig('fun.png')



# %%
# x0 = [3.14, 3.14]
# print('------------------ BFGS ------------------')
# res = opt.minimize(fun, x0, method='BFGS', jac=grad, options={'disp': True})
# print('opt res:', res.x)
# print('opt res Hessian:', res.hess_inv)
# print('\n\n\n')

# x0 = [6.28, 6.28]
# print('------------------ BFGS ------------------')
# res = opt.minimize(fun, x0, method='BFGS', jac=grad, options={'disp': True})
# print('opt res:', res.x)
# print('opt res Hessian:', res.hess_inv)
# print('\n\n\n')

# x0 = [9.42, 9.42]
# print('------------------ BFGS ------------------')
# res = opt.minimize(fun, x0, method='BFGS', jac=grad, options={'disp': True})
# print('opt res:', res.x)
# print('opt res Hessian:', res.hess_inv)
# print('\n\n\n')


# # print('Initial fun and grad', global_fun(x0))
# print('*************** Global Optimization (1) ***************')
# x1 = [[3.14, 3.14], [6.28, 6.28], [9.42, 9.42]]
# minimizer_kwargs = {"method":"BFGS", "jac":True}
# global_res = opt.basinhopping(fun_and_grad, x1, minimizer_kwargs=minimizer_kwargs, niter=200)
# # print(global_res.hess_inv.shape)
# print('opt res:', global_res.x[0], global_res.x[1])
# print('opt res:', global_res.x[2], global_res.x[3])
# print('opt res:', global_res.x[4], global_res.x[5])
# print(fun([global_res.x[0], global_res.x[1]]))
# print(fun([global_res.x[2], global_res.x[3]]))
# print(fun([global_res.x[4], global_res.x[5]]))


print('\n\n\n*************** Global Optimization (2) ***************')
bounds = [(-100, 100), (-100, 100)]
minimizer_kwargs = {"method":"BFGS", "jac":grad}
# result = opt.shgo(fun, bounds, n=100, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs, options={"disp":True})
result = opt.shgo(fun, bounds, n=100, sampling_method='sobol', minimizer_kwargs=minimizer_kwargs, options={"disp":False})
# print('global opt res', result.xl)
# print('global opt fun', result.funl)
for x in result.xl:    
    print('Point: ', x)
    res = opt.minimize(fun, x, method='BFGS', jac=grad, options={'disp': False})
    print('opt res:', res.x)
    print('opt res Hessian:', res.hess_inv)
    print('\n\n\n')




# %%
