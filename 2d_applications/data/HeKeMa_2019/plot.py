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

# just change the ROOT for ex1 - ex3
ROOT = 'ex3'
name = 'test'
extract_figures_to_files = True
FIGURE_OUTPUTS = ['png']


k = 4
N = 32

E_vh, eft, eRft, norm_of_f, to_be_updated, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD, uFineLOD_pert, NWorldCoarse, NFine, a_ref, a_pert, a_trans, f_ref, f_trans = restore_all_data(ROOT, k, N, name=name)

'''
Plot errors
'''

fig = plt.figure('error',figsize=(8,4))
ax1 = fig.add_subplot(111)

to_be_updated.append(100)
complete_errors.append(complete_errors[-1])
rel_errors.append(rel_errors[-1])

# line1 = ax1.semilogy(to_be_updated, complete_errors, 'r--', label='error')
line2 = ax1.semilogy(to_be_updated, rel_errors, 'g--', label='relative error', linewidth=2)
plt.ylabel('relative error', size=14)
plt.xlabel('updates in %', size=14)
plt.ylim([0.0008,0.15])
plt.yticks(size=14)
plt.xticks(np.arange(0,110,10), size=14)
plt.grid(alpha=0.5)

plt.savefig("tikz/{}_{}_errorplot.pdf".format(ROOT,name), bbox_inches='tight', dpi=1200)

'''
Plot error indicator
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(E_vh)), E_vh)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_evh_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

np_eft = np.einsum('i,i -> i', np.ones(np.size(eft)), eft)
draw_indicator(NWorldCoarse, np_eft, original_style=True, Gridsize=N, string='eft')
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_eft_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

np_eRft = np.einsum('i,i -> i', np.ones(np.size(eRft)), eRft)
draw_indicator(NWorldCoarse, np_eRft, original_style=True, Gridsize=N, string='eRft')
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_eRft_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

'''
plot data
'''

plt.figure("Coefficient")
drawCoefficient_origin(NFine, a_ref)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_original.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)


plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, a_pert)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_perturbed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

plt.figure('transformed')
drawCoefficient_origin(NFine, a_trans, transformed=True)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_transformed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=1200)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_rhs.{}".format(ROOT, name,fmt), bbox_inches='tight', dpi=1200)

plt.figure('Right hand side trans')
draw_f(NFine+1, f_trans)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_rhs_trans.{}".format(ROOT, name,fmt), bbox_inches='tight', dpi=1200)


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

fig = plt.figure('Solution to transformed problem')
ax = fig.add_subplot(111)
ax.imshow(np.reshape(uFineLOD, np.array([NFine[0],NFine[1]])+1), origin='lower_left', cmap= cm.hot_r)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
