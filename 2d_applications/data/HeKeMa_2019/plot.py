import matplotlib.pyplot as plt
import numpy as np
from gridlod_on_perturbations.data import restore_data

ROOT = 'ex4/'

Nstyles = {4:'k{}', 8:'y{}', 16:'b{}', 32:'r{}', 64:'k{}', 128:'k{}'}
kstyles = {1:'{}', 2:'{}', 3:'{}', 4:'{}'}

k = 3
N = 32
epsCoarse_DM, complete_tol_DM, complete_errors, tmp_errors, TOLt, uFine, uFineLOD = restore_data(ROOT, k, N)

fig = plt.figure('average epsilon domain mapping')
ax1 = fig.add_subplot(111)

ax1.set_title('k is {} and N is {}'.format(k,N))
j = 0
for i in range(len(tmp_errors)):
    if tmp_errors[i] == 0:
        j = i + 1
    else:
        break

TOLt_tmp = TOLt[j:]
tmp_errors = tmp_errors[j:]

b = 7

# line1 = ax1.loglog(TOLt_tmp, tmp_errors, Nstyles[N].format(kstyles[k].format('--')), label='gained error')
line1 = ax1.loglog(TOLt[b:], complete_errors[b:], Nstyles[N].format(kstyles[k].format('--')), label='actual error')
plt.ylabel('Error')
plt.xlabel('TOL')
plt.legend(fontsize='small', loc = 'right')
plt.grid()

print(TOLt_tmp)
print(TOLt)
print(complete_tol_DM)

ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
ax2.semilogx(TOLt, complete_tol_DM, Nstyles[N].format(kstyles[k].format('-')), label='updates')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel('corrector updates in %')
plt.legend(fontsize='small', loc = 'center left')

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

