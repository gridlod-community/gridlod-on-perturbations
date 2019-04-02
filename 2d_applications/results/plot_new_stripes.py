import csv
import matplotlib.pyplot as plt

ROOT = 'new_tol_stripes'
eps_ranges = [0.04]
NList = [32]
kList = [3]

Nstyles = {4:'k{}', 8:'y{}', 16:'b{}', 32:'g{}', 64:'k{}', 128:'k{}'}
kstyles = {1:'{}', 2:'{}', 3:'{}', 4:'{}'}


for eps_range in eps_ranges:
    for k in kList:
        for N in NList:
            complete_tol_DM = []
            complete_tol_CL = []
            complete_errors = []
            tmp_errors = []
            TOLt = []

            f = open("{}/TOLs_k{}_H{}.txt".format(ROOT,k,N), 'r')
            reader = csv.reader(f)
            for val in reader:
                TOLt.append(float(val[0]))
            f.close()

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

            f = open('{}/{}_k{}_H{}_tmp_error.txt'.format(ROOT,eps_range,k,N), 'r')
            reader = csv.reader(f)
            for val in reader:
                tmp_errors.append(float(val[0]))
            f.close()

            fig = plt.figure('average epsilon domain mapping')
            ax1 = fig.add_subplot(111)
            #ax1.set_ylim(max(complete_errors), min(complete_errors))
            ax1.set_title('k is {} and N is {}'.format(k,N))
            j = 0
            for i in range(len(tmp_errors)):
                if tmp_errors[i] == 0:
                    j = i + 1
                else:
                    break
            TOLt_tmp = TOLt[j:]
            tmp_errors = tmp_errors[j:]
            #line1 = ax1.loglog(TOLt_tmp, tmp_errors, Nstyles[N].format(kstyles[k].format('--')), label='gained error')
            line1 = ax1.loglog(TOLt, complete_errors, Nstyles[N].format(kstyles[k].format('--')), label='actual error')
            plt.ylabel('Error')
            plt.xlabel('TOL')
            plt.legend(fontsize='small', loc = 'right')
            plt.grid()

            ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)

            ax2.semilogx(TOLt, complete_tol_DM, Nstyles[N].format(kstyles[k].format('-')), label='updates')
            #ax2.semilogx(TOL, complete_tol_CL, styles[eps_range].format('--'))
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            plt.ylabel('corrector updates in %')
            plt.legend(fontsize='small', loc = 'lower center')



plt.show()