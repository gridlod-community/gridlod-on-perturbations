import matplotlib.pyplot as plt
import numpy as np
from gridlod_on_perturbations.data import restore_all_data
from gridlod_on_perturbations.visualization_tools import drawCoefficient_origin, draw_f, draw_indicator
from matplotlib import cm
import matplotlib
# matplotlib.style.use('ggplot')
matplotlib.rc('font', family='sans-serif')
# http://nerdjusttyped.blogspot.de/2010/07/type-1-fonts-and-matplotlib-figures.html
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

# just change the ROOT for ex1 - ex3 and exOWR
ROOT = 'ex2_noisy'
name = 'test'
extract_figures_to_files = True
FIGURE_OUTPUTS = ['png']
dpi = None

k = 4
N = 32

E_vh, eft, eRft, norm_of_f, to_be_updated, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD, uFineLOD_pert, NWorldCoarse, NFine, a_ref, a_pert, a_trans, f_ref, f_trans = restore_all_data(ROOT, k, N, name=name)

'''
Plot errors
'''

fig = plt.figure('error')
ax1 = fig.add_subplot(111)
to_be_updated.append(100)
complete_errors.append(0)
rel_errors.append(0)
TOLt.insert(0,TOLt[0]+1e-5)
x_points = np.linspace(0,100,100000)
yid = [1e2-x * (1e2 - 1e-5)/100 for x in x_points]
# ax1.semilogy(x_points, yid, alpha=0.1)
line1 = ax1.semilogy(to_be_updated, rel_errors, 'g', label='$\mathcal{E}_{rel}$', linewidth=1)
ax1.semilogy(to_be_updated, TOLt, 'k--', label='TOL', alpha=0.5)
# ax1.set_yticks([pow(10,i) for i in range(-7,3)])
print(TOLt)
ax1.set_yticks([pow(10,i) for i in range(-6,3)])
# ax1.set_yscale('log', basey=10, )
plt.xlabel('updates in %', size=16)
# ax1.set_yscale('log', subsy=[2,3,4,5,6,7,8,9], basey=10)
ax1.set_ylim([3e-7,2e2])
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

plt.xticks(np.arange(0,110,10), size=16)
# plt.yticks(size=16)
plt.grid(alpha=0.5)



ax1.legend(loc='upper right', fontsize=16)


plt.savefig("tikz/{}_{}_errorplot.pdf".format(ROOT,name), bbox_inches='tight', dpi=dpi)

plt.show()
'''
Plot error indicator
'''
np_eps = np.einsum('i,i -> i', np.ones(np.size(E_vh)), E_vh)
draw_indicator(NWorldCoarse, np_eps, original_style=True, Gridsize=N)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_evh_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)

np_eft = np.einsum('i,i -> i', np.ones(np.size(eft)), eft)
draw_indicator(NWorldCoarse, np_eft, original_style=True, Gridsize=N, string='eft')
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_eft_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)

np_eRft = np.einsum('i,i -> i', np.ones(np.size(eRft)), eRft)
draw_indicator(NWorldCoarse, np_eRft, original_style=True, Gridsize=N, string='eRft')
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_eRft_indicator.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)

'''
plot data
'''

plt.figure("Coefficient")
if ROOT == "ex2_noisy":
    lim = [np.max(a_trans),np.min(a_trans)]   # for ex2
elif ROOT == "ex3_noisy":
    lim = [np.max(a_trans),np.min(a_ref)]   # for ex3
elif ROOT == "ex2":
    lim = [np.max(a_trans),np.min(a_trans)]   # for ex2
elif ROOT == "ex3":
    lim = [np.max(a_trans),np.min(a_ref)]   # for ex3
else:
    lim = None
drawCoefficient_origin(NFine, a_ref, lim=lim)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_original.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)


plt.figure("Perturbed coefficient")
drawCoefficient_origin(NFine, a_pert, lim=lim)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_perturbed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)

plt.figure('transformed')
drawCoefficient_origin(NFine, a_trans, transformed=True, lim=lim)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_transformed.{}".format(ROOT,name,fmt), bbox_inches='tight', dpi=dpi)

plt.figure('Right hand side')
draw_f(NFine+1, f_ref)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_rhs.{}".format(ROOT, name,fmt), bbox_inches='tight', dpi=dpi)

plt.figure('Right hand side trans')
draw_f(NFine+1, f_trans)
if extract_figures_to_files:
    for fmt in FIGURE_OUTPUTS:
        plt.savefig("tikz/{}_{}_rhs_trans.{}".format(ROOT, name,fmt), bbox_inches='tight', dpi=dpi)


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

# plt.show()
