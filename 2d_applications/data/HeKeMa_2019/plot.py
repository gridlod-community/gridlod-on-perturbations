import matplotlib.pyplot as plt
import numpy as np
from gridlod_on_perturbations.data import restore_all_data

from visualization_tools import drawCoefficient_origin, draw_f, draw_indicator

# just change the ROOT for ex1 - ex4
ROOT = 'ex4'


k = 4
N = 32

epsCoarse, to_be_updated, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD, NWorldCoarse, NFine, a_ref, a_pert, a_trans, f_ref, f_trans = restore_all_data(ROOT, k, N)

'''
Plot errors
'''

fig = plt.figure('error')
ax1 = fig.add_subplot(111)

to_be_updated.append(100)
complete_errors.append(complete_errors[-1])
rel_errors.append(rel_errors[-1])

line1 = ax1.semilogy(to_be_updated, complete_errors, 'r--', label='error')
line2 = ax1.semilogy(to_be_updated, rel_errors, 'g--', label='relative error')
plt.ylabel('Error')
plt.xlabel('Updates in %')
plt.legend(fontsize='small', loc = 'right')
plt.ylim([0.001,1])
plt.xticks(np.arange(0,110,10))
plt.grid()

'''
Plot error indicator
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(epsCoarse)), epsCoarse)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)

'''
plot data
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, a_ref)

plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, a_pert)

plt.figure('transformed')
drawCoefficient_origin(NFine, a_trans)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)

'''
Plot solutions
'''


fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFine, np.array([NFine[0],NFine[1]])+1), origin='lower_left')

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineLOD, np.array([NFine[0],NFine[1]])+1), origin='lower_left')


plt.show()

