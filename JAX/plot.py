import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

var_num = [12, 24, 36, 48, 60, 72, 84, 96, 108, 360]
exec_time = [0.33732056617736816, 0.5926854610443115, 0.858231782913208, 1.0935347080230713, 1.4335558414459229, 1.703127145767212, 
            1.8849246501922607, 2.1099965572357178, 2.4226298332214355, 8.073891639709473]

plt.plot(var_num, exec_time)
plt.xlabel('Number of Variables')
plt.ylabel('Execution time')
plt.savefig('num-time_graph.png')