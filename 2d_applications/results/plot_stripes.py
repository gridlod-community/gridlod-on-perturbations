import csv
import matplotlib.pyplot as plt

ROOT = 'stripes/'
eps_ranges = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

TOL = []
f = open("%s/TOLs.txt" % ROOT, 'r')
reader = csv.reader(f)
for val in reader:
        TOL.append(float(val[0]))
f.close()

for eps_range in eps_ranges:
    complete_tol_DM = []

    f = open('{}/{}.txt'.format(ROOT,eps_range), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_tol_DM.append(float(val[0]))
    f.close()

    plt.figure('average epsilon')
    plt.title('Average updates for various' + r' $\varepsilon$')
    plt.semilogx(TOL, complete_tol_DM, label='%8.5f' % eps_range)
    plt.xlabel('TOL')
    plt.ylabel('Updated in %')
    #plt.semilogx(TOL, 1 / MC * complete_tol_CL, label='classic')
    plt.legend(fontsize='small')

plt.show()