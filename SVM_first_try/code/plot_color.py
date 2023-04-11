import numpy as np
import matplotlib.pyplot as plt

dims_array = np.load('dims_array.npy')
C_param_array = np.load('C_param_array.npy')
gamma_param_array = np.load('gamma_param_array.npy')
errorresult2 = np.load('errorresult2.npy')
errorresult6 = np.load('errorresult6.npy')

for i, dims in enumerate(dims_array):
    plt.figure()
    plt.title(f"Binary, {dims} dimensions")
    plt.contourf(C_param_array, gamma_param_array, errorresult2[i])
    plt.axis('scaled')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("C")
    plt.ylabel("gamma")    
    plt.colorbar()

    plt.figure()
    plt.title(f"6-class, {dims} dimensions")
    plt.contourf(C_param_array, gamma_param_array, errorresult6[i])
    plt.axis('scaled')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.colorbar()

plt.show()