import matplotlib.pyplot as plt
import numpy as np
from gridlod_on_perturbations.data import restore_all_data
import matplotlib2tikz
from visualization_tools import drawCoefficient_origin, draw_f, draw_indicator
from matplotlib import cm
import matplotlib
# matplotlib.style.use('ggplot')
matplotlib.rc('font', family='sans-serif')
# http://nerdjusttyped.blogspot.de/2010/07/type-1-fonts-and-matplotlib-figures.html
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

# just change the ROOT for ex1 - ex4
ROOT = 'ex5'
name = 'check_errors'
extract_figures_to_files = False
FIGURE_OUTPUTS = ['png', 'eps']


k = 3
N = 16

epsCoarse, _, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD, NWorldCoarse, NFine, a_ref, a_pert, a_trans, f_ref, f_trans, Eft = restore_all_data(ROOT, k, N, name=name)
_ , to_be_updated , _, _, rel_errors_reverse, _, _, _, _, _, _, _, _, _, _, _ = restore_all_data(ROOT, k, N, name='check_errors_no_Eft')

'''
Plot errors
'''

fig = plt.figure('error')
ax1 = fig.add_subplot(111)
to_be_updated.append(10)
rel_errors.append(rel_errors[-1])
rel_errors_reverse.append(rel_errors_reverse[-1])

line1 = ax1.semilogy(to_be_updated, rel_errors, 'g--', label='update_due_to_Eft', linewidth=2)
line2 = ax1.semilogy(to_be_updated, rel_errors_reverse, 'r--', label='update_first_the_dotted_area', linewidth=2)
plt.ylabel('relative error')
plt.xlabel('updates in %')
plt.ylim([0.01,1])
plt.xticks(np.arange(0,12,2))
plt.grid(alpha=0.5)
plt.legend()

matplotlib2tikz.save("tikz/{}_{}_errorplot.tex".format(ROOT,name))

'''
Plot error indicator
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(Eft)), Eft)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)
'''
plot data
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, a_ref)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_original.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)


# plt.figure("Perturbed coefficient")
# drawCoefficient_origin(NFine, a_pert)
# if extract_figures_to_files:
#     for fmt in FIGURE_OUTPUTS:
#         plt.savefig("tikz/{}_{}_perturbed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)
#
# plt.figure('transformed')
# drawCoefficient_origin(NFine, a_trans)
# if extract_figures_to_files:
#     for fmt in FIGURE_OUTPUTS:
#         plt.savefig("tikz/{}_{}_transformed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_rhs.{}".format(ROOT, name,fmt), bbox_inches='tight', dpi=1200)


'''
Plot solutions
'''
fig = plt.figure('Solutions')
ax = fig.add_subplot(121)
ax.set_title('Fem Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFine, np.array([NFine[0],NFine[1]])+1), origin='lower_left', cmap= cm.hot_r)
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(122)
ax.set_title('PGLOD Solution to transformed problem',fontsize=6)
ax.imshow(np.reshape(uFineLOD, np.array([NFine[0],NFine[1]])+1), origin='lower_left', cmap= cm.hot_r)
ax.set_xticks([])
ax.set_yticks([])

# fig = plt.figure('Solution to transformed problem')
# ax = fig.add_subplot(111)
# ax.imshow(np.reshape(uFineLOD, np.array([NFine[0],NFine[1]])+1), origin='lower_left', cmap= cm.hot_r)
# ax.set_xticks([])
# ax.set_yticks([])


plt.show()

