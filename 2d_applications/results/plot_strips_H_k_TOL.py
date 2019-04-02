import csv
import matplotlib.pyplot as plt

ROOT = 'stripes_H_k_TOL/'
eps_ranges = [0.04]
NList = [8,32,128]
kList = [3]

Nstyles = {4:'k{}', 8:'y{}', 16:'b{}', 32:'r{}', 64:'k{}', 128:'k{}'}
kstyles = {1:'{}', 2:'{}', 3:'{}', 4:'{}'}
TOL = []
f = open("%s/TOLs.txt" % ROOT, 'r')
reader = csv.reader(f)
for val in reader:
        TOL.append(float(val[0]))
f.close()

for eps_range in eps_ranges:
    for k in kList:
        for N in NList:
            complete_tol_DM = []
            complete_tol_CL = []
            complete_errors = []

            f = open('{}/{}_k{}_H{}_DM.txt'.format(ROOT,eps_range,k,N), 'r')
            reader = csv.reader(f)
            for val in reader:
                complete_tol_DM.append(float(val[0]))
            f.close()

            f = open('{}/{}_k{}_H{}_CL.txt'.format(ROOT,eps_range,k,N), 'r')
            reader = csv.reader(f)
            for val in reader:
                complete_tol_CL.append(float(val[0]))
            f.close()

            f = open('{}/{}_k{}_H{}_error.txt'.format(ROOT,eps_range,k,N), 'r')
            reader = csv.reader(f)
            for val in reader:
                complete_errors.append(float(val[0]))
            f.close()

            fig = plt.figure('average epsilon domain mapping')
            ax1 = fig.add_subplot(111)
            #ax1.set_ylim(max(complete_errors), min(complete_errors))
            #ax1.set_title('Average classic updates for various' + r' $\varepsilon$')

            line1 = ax1.loglog(TOL, complete_errors, Nstyles[N].format(kstyles[k].format('--')))
            plt.ylabel('Error')
            plt.xlabel('TOL')

            ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)

            ax2.semilogx(TOL, complete_tol_DM, Nstyles[N].format(kstyles[k].format('-')), label='k:{}, N:{}'.format(k,N))
            #ax2.semilogx(TOL, complete_tol_CL, styles[eps_range].format('--'))
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            plt.ylabel('corrector updates in %')
            plt.legend(fontsize='small')


plt.show()