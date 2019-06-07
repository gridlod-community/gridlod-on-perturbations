import csv

def store_all_data(ROOT, k, N, epsCoarse_DM, to_be_updatedT_DM, energy_errorT, tmp_errorT, TOLt, uFine, uFineLOD,
                   NWorldFine, NWorldCoarse, ABase, APert, f_ref, ATrans = None, f_trans = None, name='test'):
    if ATrans is None:
        ATrans = APert
    if f_trans is None:
        f_trans = f_ref

    with open('{}/{}_k{}_H{}_epsCoarse.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in epsCoarse_DM:
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


def store_minimal_data(ROOT, k, N, epsCoarse_DM, to_be_updatedT_DM, energy_errorT, tmp_errorT, TOLt, uFine,
                       uFineLOD, name = 'test'):
    with open('{}/{}_k{}_H{}_epsCoarse.txt'.format(ROOT, name, k, N), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for val in epsCoarse_DM:
            writer.writerow([val])

    with open('{}/{}_k{}_H{}_to_be_updated.txt'.format(ROOT ,name, k, N), 'w') as csvfile:
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


def restore_all_data(ROOT, k, N, name = 'test'):
    epsCoarse_DM = []
    complete_tol_DM = []
    complete_errors = []
    tmp_errors = []
    TOLt = []
    uFine = []
    uFineLOD = []
    NWorldCoarse = []
    NWorldFine = []
    a_ref = []
    a_pert = []
    a_trans = []
    f_ref = []
    f_trans = []

    f = open('{}/{}_k{}_H{}_epsCoarse.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        epsCoarse_DM.append(float(val[0]))
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

    f = open("{}/{}_NWorldFine_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        NWorldFine.append(float(val[0]))
    f.close()

    #safe NworldCoarse
    f = open("{}/{}_NWorldCoarse_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        NWorldCoarse.append(float(val[0]))
    f.close()

    #ABase
    f = open("{}/{}_OriginalCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_ref.append(float(val[0]))
    f.close()

    # APert
    f = open("{}/{}_PerturbedCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_pert.append(float(val[0]))
    f.close()

    f = open("{}/{}_TransformedCoeff_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        a_trans.append(float(val[0]))
    f.close()

    # APert
    f = open("{}/{}_f_ref_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        f_ref.append(float(val[0]))
    f.close()

    f = open("{}/{}_f_trans_k{}_H{}.txt".format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        f_trans.append(float(val[0]))
    f.close()

    return epsCoarse_DM, complete_tol_DM, complete_errors, tmp_errors, TOLt, uFine, uFineLOD, NWorldCoarse, NWorldFine, a_ref, a_pert, a_trans, f_ref, f_trans


def restore_minimal_data(ROOT, k, N, name = 'test'):
    epsCoarse_DM = []
    to_be_updated = []
    complete_errors = []
    tmp_errors = []
    TOLt = []
    uFine = []
    uFineLOD = []

    f = open('{}/{}_k{}_H{}_epsCoarse.txt'.format(ROOT, name, k, N), 'r')
    reader = csv.reader(f)
    for val in reader:
        epsCoarse_DM.append(float(val[0]))
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


    return epsCoarse_DM, to_be_updated, complete_errors, tmp_errors, TOLt, uFine, uFineLOD