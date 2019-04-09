import csv
import matplotlib.pyplot as plt

ROOT = 'local_bending_stripes'
eps_ranges = [0]
NList = [32]
kList = [2]

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

            b = 5
            e = -1
            #line1 = ax1.loglog(TOLt_tmp, tmp_errors, Nstyles[N].format(kstyles[k].format('--')), label='gained error')
            line1 = ax1.loglog(TOLt[b:], complete_errors[b:], Nstyles[N].format(kstyles[k].format('--')), label='actual error')
            plt.ylabel('Error')
            plt.xlabel('TOL')
            plt.legend(fontsize='small', loc = 'right')
            plt.grid()

            ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
            ax2.semilogx(TOLt[b:], complete_tol_DM[b:], Nstyles[N].format(kstyles[k].format('-')), label='updates')
            #ax2.semilogx(TOL, complete_tol_CL, styles[eps_range].format('--'))
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            plt.ylabel('corrector updates in %')
            plt.legend(fontsize='small', loc = 'center left')



plt.show()

if 0:
    # Just for the picture

    import numpy as np

    from gridlod import util, func

    from MasterthesisLOD import buildcoef2d
    from gridlod_on_perturbations import discrete_mapping
    from visualization_tools import drawCoefficient_origin

    factor = 8

    fine = 256 * factor
    NFine = np.array([fine,fine])
    NpFine = np.prod(NFine + 1)

    space = 32 * factor
    thick = 4 * factor

    bg = 0.1		#background
    val = 1			#values

    CoefClass = buildcoef2d.Coefficient2d(NFine,
                            bg                  = bg,
                            val                 = val,
                            length              = 1,
                            thick               = thick,
                            space               = space,
                            probfactor          = 1,
                            right               = 1,
                            down                = 0,
                            diagr1              = 0,
                            diagr2              = 0,
                            diagl1              = 0,
                            diagl2              = 0,
                            LenSwitch           = None,
                            thickSwitch         = None,
                            equidistant         = True,
                            ChannelHorizontal   = None,
                            ChannelVertical     = True,
                            BoundarySpace       = True)


    #global variables
    global aFine_ref
    global aFine_trans
    global aFine_pert
    global KmsijT
    global correctorsListT
    global f_trans

    # Set reference coefficient
    aFine_ref_shaped = CoefClass.BuildCoefficient()
    aFine_ref_shaped = CoefClass.SpecificMove(Number=np.arange(0,10), steps=4, Right=1)
    aFine_ref = aFine_ref_shaped.flatten()
    number_of_channels = len(CoefClass.ShapeRemember)

    f_pert = np.ones(NpFine)

    size_of_an_element = 1./fine
    print('the size of a fine element is {}'.format(size_of_an_element))
    walk_with_perturbation = size_of_an_element

    channels_position_from_zero = space
    channels_end_from_zero = channels_position_from_zero + thick

    xpFine = util.pCoordinates(NFine)
    xtFine = util.tCoordinates(NFine)

    #I want to know the exact places of the channels
    ref_array = aFine_ref_shaped[0]



    def create_psi_function():
        global aFine_pert
        global f_trans
        epsilonT = []

        forward_mapping = np.stack([xpFine[:, 0], xpFine[:, 1]], axis=1)

        xpFine_shaped = xpFine.reshape(fine + 1, fine + 1, 2)
        left, right = 0, fine + 1

        for c in range(number_of_channels):
            count = 0
            for i in range(np.size(ref_array)):
                if ref_array[i] == 1:
                    count +=1
                if count == (c+1)*thick:
                    begin = i + 1 - space // 2
                    end = i + 1 + thick+ space // 2
                    break
            print(begin,end)
            left_2, right_2 = begin, end
            if c == 3:
                epsilon = 25
            #elif c == 4:
            #    epsilon = -25
            else:
                epsilon = np.random.uniform(-15,15)

            part_x = xpFine_shaped[left:right, left_2:right_2, 0]
            part_y = xpFine_shaped[left:right, left_2:right_2, 1]
            left_margin_x = np.min(part_x)
            right_margin_x = np.max(part_x)
            left_margin_y = np.min(part_y)
            right_margin_y = np.max(part_y)

            print(left_margin_x, right_margin_x, left_margin_y, right_margin_y)

            forward_mapping_partial = np.stack([xpFine_shaped[left:right, left_2:right_2, 0]
                                                + epsilon *
                                                (xpFine_shaped[left:right, left_2:right_2, 0] - left_margin_x) *
                                                (right_margin_x - xpFine_shaped[left:right, left_2:right_2, 0]) *
                                                (xpFine_shaped[left:right, left_2:right_2, 1] - left_margin_y) *
                                                (right_margin_y - xpFine_shaped[left:right, left_2:right_2, 1]),
                                                xpFine_shaped[left:right, left_2:right_2, 1]], axis=2)

            forward_mapping_shaped = forward_mapping.reshape(fine + 1, fine + 1, 2)
            forward_mapping_shaped[left:right, left_2:right_2, :] = forward_mapping_partial

            epsilonT.append(epsilon)

        forward_mapping = forward_mapping_shaped.reshape((fine + 1) ** 2, 2)


        print('Those are the results of the shift epsilon', epsilonT)

        psi = discrete_mapping.MappingCQ1(NFine, forward_mapping)

        aFine_ref = aFine_ref_shaped.flatten()

        xtFine_pert = psi.evaluate(xtFine)
        xtFine_ref = psi.inverse_evaluate(xtFine)
        xpFine_pert = psi.evaluate(xpFine)
        xpFine_ref = psi.inverse_evaluate(xpFine)

        aFine_pert = func.evaluateDQ0(NFine, aFine_ref, xtFine_ref)
        aBack_ref = func.evaluateDQ0(NFine, aFine_pert, xtFine_pert)

        f_ref = func.evaluateCQ1(NFine, f_pert, xpFine_pert)
        f_trans = np.einsum('t, t -> t', f_ref, psi.detJ(xpFine))

        return psi


    psi = create_psi_function()
    plt.figure("Coefficient")
    drawCoefficient_origin(NFine, aFine_ref)

    plt.figure("a_perturbed")
    drawCoefficient_origin(NFine, aFine_pert)

    plt.show()