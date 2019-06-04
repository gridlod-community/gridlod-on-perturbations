import matplotlib.pyplot as plt
import numpy as np
from gridlod_on_perturbations.data import restore_minimal_data

ROOT = '20x20dots/'

Nstyles = {4:'k{}', 8:'y{}', 16:'b{}', 32:'r{}', 64:'k{}', 128:'k{}'}
kstyles = {1:'{}', 2:'{}', 3:'{}', 4:'{}'}

k = 4
N = 32
epsCoarse_DM, to_be_updated, complete_errors, tmp_errors, TOLt, uFine, uFineLOD = restore_minimal_data(ROOT, k, N, name='test2_perc')

fig = plt.figure('average epsilon domain mapping')
ax1 = fig.add_subplot(111)

ax1.set_title('k is {} and N is {}'.format(k,N))

to_be_updated.append(100)
complete_errors.append(complete_errors[-1])

line1 = ax1.semilogy(to_be_updated, complete_errors, Nstyles[N].format(kstyles[k].format('--')), label='actual error')
plt.ylabel('Error')
plt.xlabel('Updates in %')
plt.legend(fontsize='small', loc = 'right')
plt.grid()



'''
Plot solutions
'''

fine = 256
NFine = np.array([fine,fine])

fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFine, NFine+1), origin='lower_left')

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineLOD, NFine+1), origin='lower_left')

plt.show()

