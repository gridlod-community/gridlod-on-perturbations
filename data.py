import csv
import numpy as np

def store_all_data(ROOT, k, N, E_vh_DM, np_eft, np_eRft, norm_of_f, to_be_updatedT_DM, energy_errorT, tmp_errorT, rel_energy_errorT, TOLt, uFine, uFineLOD,
                   NWorldFine, NWorldCoarse, ABase, APert, f_ref, ATrans = None, f_trans = None, uFineLOD_pert = None, name='test'):
    if ATrans is None:
        ATrans = APert
    else:
        try:
            ATrans = np.linalg.norm(ATrans, axis=(1, 2), ord=2)
        except:
            pass

    if f_trans is None:
        f_trans = f_ref

    if uFineLOD_pert is None:
        uFineLOD_pert = uFineLOD

    with open('{}/{}_k{}_H{}_E_vh.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in E_vh_DM:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_eft.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in np_eft:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_eRft.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in np_eRft:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_norm_of_f.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in norm_of_f:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_to_be_updated.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in to_be_updatedT_DM:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_error.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in energy_errorT:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_tmp_error.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in tmp_errorT:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_rel_error.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in rel_energy_errorT:
            writer.writerow([val])

    with open('{}/{}_TOLs_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in TOLt:
            writer.writerow([val])

    with open('{}/{}_uFine_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in uFine:
            writer.writerow([val])

    with open('{}/{}_uFineLOD_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in uFineLOD:
            writer.writerow([val])

    with open('{}/{}_uFineLOD_pert_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in uFineLOD_pert:
            writer.writerow([val])

    with open('{}/{}_NWorldFine_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in NWorldFine:
            writer.writerow([val])

    #safe NworldCoarse
    with open('{}/{}_NWorldCoarse_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in NWorldCoarse:
            writer.writerow([val])

    #ABase
    with open('{}/{}_OriginalCoeff_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in ABase:
            writer.writerow([val])

    # APert
    with open('{}/{}_PerturbedCoeff_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in APert:
            writer.writerow([val])

    # APert
    with open('{}/{}_TransformedCoeff_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in ATrans:
            writer.writerow([val])

    # APert
    with open('{}/{}_f_ref_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in f_ref:
            writer.writerow([val])

    # APert
    with open('{}/{}_f_trans_k{}_H{}.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in f_trans:
            writer.writerow([val])


def restore_all_data(ROOT, k, N, name = 'test'):
    E_vh_DM = []
    eft = []
    eRft = []
    norm_of_f = []
    complete_tol_DM = []
    complete_errors = []
    tmp_errors = []
    rel_errors = []
    TOLt = []
    uFine = []
    uFineLOD = []
    uFineLOD_pert = []
    NWorldCoarse = np.array([])
    NWorldFine = np.array([])
    a_ref = np.array([])
    a_pert = np.array([])
    a_trans = np.array([])
    f_ref = np.array([])
    f_trans = np.array([])

    f = open('{}/{}_k{}_H{}_E_vh.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        E_vh_DM.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_eft.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        eft.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_eRft.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        eRft.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_norm_of_f.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        norm_of_f.append(float(val[0]))
    f.close()


    f = open("{}/{}_TOLs_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        TOLt.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_to_be_updated.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_tol_DM.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_errors.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_tmp_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        tmp_errors.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_rel_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        rel_errors.append(float(val[0]))
    f.close()

    f = open("{}/{}_uFine_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        uFine.append(float(val[0]))
    f.close()

    f = open("{}/{}_uFineLOD_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        uFineLOD.append(float(val[0]))
    f.close()

    try:
        f = open("{}/{}_uFineLOD_pert_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
        reader = csv.reader(f)
        for val in reader:
            uFineLOD_pert.append(float(val[0]))
        f.close()
    except:
        pass

    f = open("{}/{}_NWorldFine_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        NWorldFine = np.append(NWorldFine, [np.int(val[0])])
    NWorldFine = NWorldFine.reshape([1, 2])[0].astype(int)
    f.close()

    #safe NworldCoarse
    f = open("{}/{}_NWorldCoarse_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        NWorldCoarse = np.append(NWorldCoarse, [int(val[0])])
    NWorldCoarse = NWorldCoarse.reshape([1,2])[0].astype(int)
    f.close()

    #ABase
    f = open("{}/{}_OriginalCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_ref = np.append(a_ref, [float(val[0])])
    f.close()

    # APert
    f = open("{}/{}_PerturbedCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_pert = np.append(a_pert, [float(val[0])])
    f.close()

    f = open("{}/{}_TransformedCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_trans = np.append(a_trans, [float(val[0])])
    f.close()

    # APert
    f = open("{}/{}_f_ref_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        f_ref = np.append(f_ref, [float(val[0])])
    f.close()

    f = open("{}/{}_f_trans_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        f_trans = np.append(f_trans, [float(val[0])])
    f.close()

    return E_vh_DM, eft, eRft, norm_of_f, complete_tol_DM, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD, uFineLOD_pert, NWorldCoarse, NWorldFine, a_ref, a_pert, a_trans, f_ref, f_trans


def restore_minimal_data(ROOT, k, N, name = 'test'):
    E_vh_DM = []
    to_be_updated = []
    complete_errors = []
    tmp_errors = []
    rel_errors = []
    TOLt = []
    uFine = []
    uFineLOD = []

    f = open('{}/{}_k{}_H{}_E_vh.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        E_vh_DM.append(float(val[0]))
    f.close()

    f = open("{}/{}_TOLs_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        TOLt.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_to_be_updated.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        to_be_updated.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        complete_errors.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_tmp_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        tmp_errors.append(float(val[0]))
    f.close()

    f = open('{}/{}_k{}_H{}_rel_error.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        rel_errors.append(float(val[0]))
    f.close()

    f = open("{}/{}_uFine_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        uFine.append(float(val[0]))
    f.close()

    f = open("{}/{}_uFineLOD_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        uFineLOD.append(float(val[0]))
    f.close()

    return E_vh_DM, to_be_updated, complete_errors, tmp_errors, rel_errors, TOLt, uFine, uFineLOD