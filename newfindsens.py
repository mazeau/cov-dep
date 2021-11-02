import os
import pandas as pd
import numpy as np
import multiprocessing

def import_sensitivities(ratio, file_location):
    """
    Ratio is the C/O starting gas ratio
    file_location is the LSR C and O binding energy, fasle to load the base case
    """
    tol_list = ['10_16_', '10_17_', '10_18_', '10_19_', '10_20_', '10_21_',
                '10_22_', '10_23_', '10_24_', '11_16_', '11_17_', '11_18_',
                '11_19_', '11_20_', '11_21_', '11_22_', '11_23_', '11_24_',
                '12_16_', '12_17_', '12_18_', '12_19_', '12_20_', '12_21_',
                '12_22_', '12_23_', '12_24_',]

    fail_count = 0
    # load_count = 0
    for tol in tol_list:
        try:
            pd.read_csv('newdata/' + file_location + '/' + tol + str(ratio) + 'RxnSensitivity.csv')
            print('found ' + file_location + '/' + str(ratio) + 'RxnSensitivity.csv')
            return
        except:
            fail_count += 1

    if fail_count == len(tol_list):
        print('Cannot find sensitivity for ' + file_location + '/' + str(ratio) + 'RxnSensitivity.csv')


array = os.listdir('./linearscaling/')
array = sorted(array)
max_cpus = multiprocessing.cpu_count()
ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
# tol_list = ['10_16_', '10_17_', '10_18_', '10_19_', '10_20_', '10_21_',
#             '10_22_', '10_23_', '10_24_', '11_16_', '11_17_', '11_18_',
#             '11_19_', '11_20_', '11_21_', '11_22_', '11_23_', '11_24_',
#             '12_16_', '12_17_', '12_18_', '12_19_', '12_20_', '12_21_',
#             '12_22_', '12_23_', '12_24_',]
# get all combinations of the above for importing
# inputs = list(itertools.product(tol_list, ratios))

# allrxndata = []  # where all rxn sens itivities will be stored
# allthermodata = []  # where all thermo sensitivities will be stored

# for f in array:
#     # rxndata = []
#     # thermodata = []
#     for ratio in ratios:
#         import_sensitivities(ratio, f)
        # thermodata.append(import_sensitivities(ratio, file_location=f, thermo=True))
    # allrxndata.append(rxndata)
    # allthermodata.append(thermodata)


def loadSensDataWorker(f_location):
    ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
    for ratio in ratios:
        import_sensitivities(ratio, f_location)


num_threads = max_cpus
# lump = (81. / max_cpus) + 1
lump = 1
pool = multiprocessing.Pool(processes=num_threads)
pool.map(loadSensDataWorker, array, lump)
pool.close()
pool.join()
