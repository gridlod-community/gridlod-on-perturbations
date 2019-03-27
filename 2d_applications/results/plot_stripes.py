import csv
import matplotlib.pyplot as plt

ROOT = 'stripes/'
eps_ranges = [0.007, 0.01, 0.03]
#eps_ranges = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03]
styles = {0.003:'k{}', 0.007:'y{}', 0.01:'b{}', 0.03:'r{}'}
TOL = []
f = open("%s/TOLs.txt" % ROOT, 'r')
reader = csv.reader(f)
for val in reader:
        TOL.append(float(val[0]))
f.close()

for eps_range in eps_ranges:
    complete_tol_DM = []
    complete_tol_CL = []
    complete_errors = []

    f = open('{}/{}_DM.txt'.format(ROOT,eps_range), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_tol_DM.append(float(val[0]))
    f.close()

    f = open('{}/{}_CL.txt'.format(ROOT,eps_range), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_tol_CL.append(float(val[0]))
    f.close()

    f = open('{}/{}_error.txt'.format(ROOT,eps_range), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_errors.append(float(val[0]))
    f.close()

    fig = plt.figure('average epsilon domain mapping')
    ax1 = fig.add_subplot(111)
    #ax1.set_ylim(max(complete_errors), min(complete_errors))
    #ax1.set_title('Average classic updates for various' + r' $\varepsilon$')

    line1 = ax1.loglog(TOL, complete_errors, styles[eps_range].format('--'))
    plt.ylabel('Error')
    plt.xlabel('TOL')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)

    ax2.semilogx(TOL, complete_tol_DM, styles[eps_range].format('-'), label='%8.5f' % eps_range)
    #ax2.semilogx(TOL, complete_tol_CL, styles[eps_range].format('--'))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('corrector updates in %')
    plt.legend(fontsize='small')


plt.show()