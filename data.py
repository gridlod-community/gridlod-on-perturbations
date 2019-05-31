import csv

def safe_data(ROOT, k, N, epsCoarse_DM, to_be_updatedT_DM, energy_errorT, tmp_errorT, TOLt, uFine, uFineLOD, name = 'test'):
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


def restore_data(ROOT, k, N, name = 'test'):
    epsCoarse_DM = []
    complete_tol_DM = []
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


    return epsCoarse_DM, complete_tol_DM, complete_errors, tmp_errors, TOLt, uFine, uFineLOD