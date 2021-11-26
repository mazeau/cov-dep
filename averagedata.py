import os
import re
import pandas as pd
import sys
import multiprocessing
import itertools
import statistics
import numpy as np


max_cpus = multiprocessing.cpu_count()

array = os.listdir('small-grid/')
array = sorted(array)
files = ['./small-grid/' + x for x in array]
array = os.listdir('small-grid-cov/')
array = sorted(array)
allfiles = files + ['./small-grid-cov/' + x for x in array] + ['./pt_cathub/','./pt_cathub_cov/']

ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]

rtol = np.arange(9,13)
atol = np.arange(16,25)

tol_list = ['{:02d}_{:02d}_'.format(x, y) for x in rtol for y in atol]


def import_data(tol, file_location):
    """
    Import raw data from simulations.
    """
    try:
        data = pd.read_csv(file_location + '/all-data/' + tol + 'data.csv')

        data = data.values
        data = data.tolist()
        return data
    except:
        print('Cannot find ' + file_location + '/all-data/' + tol + 'data.csv')

def filter_outliers(data, verbose=False):
    """
    Filters out outliers using the IQR.
    """
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * 2
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]

    if verbose is True:
        if len(outliers) > 0:
            print(f"Removing {len(outliers)} outliers")

    return outliers_removed


def calculate(data):
    ratio, ch4_in, ch4_out, co_out, h2_out, h2o_out, co2_out, exit_T, max_T, dist_Tmax, o2_conv, max_ch4_conv, dist_50_ch4_conv = data

    ch4_depletion = ch4_in - ch4_out
    ch4_conv = ch4_depletion / ch4_in
    h2_sel = h2_out / (ch4_depletion * 2)
    h2_yield = h2_out / ( ch4_in * 2)
    co_sel = co_out / ch4_depletion
    co_yield = co_out / ch4_in
    syngas_sel = co_sel + h2_sel
    syngas_yield = syngas_sel * ch4_conv
    co2_sel = co2_out / ch4_depletion
    h2o_sel = h2o_out / (2 * ch4_depletion)
    fullox_sel = h2o_sel + co2_sel
    fullox_yield = fullox_sel * ch4_conv

    return ratio, syngas_sel, syngas_yield, co_sel, co_yield, h2_sel, h2_yield, ch4_conv, fullox_sel, fullox_yield, exit_T, max_T, dist_Tmax, o2_conv


def average_data(data, type='avg', verbose=False):
    """
    Average raw simulation data together, without checking for outliers.
    """
    tol = len(data)
    ratio = len(data[0])
    value = len(data[0][0])

    out = []
    out_var = []
    for r in range(ratio):
        fixed_data = []
        var_data = []
        for s in range(1, value):
            tmp_list = []
            for t in range(tol):
                tmp_list.append(data[t][r][s])

            new_data = filter_outliers(tmp_list)

            avg = statistics.mean(new_data)
            var = statistics.variance(new_data)

            if s is 1:
                avg = round(avg,1)
                var = round(avg,1)

            fixed_data.append(avg)
            var_data.append(var)
        out.append(list(calculate(fixed_data)))
        out_var.append(var_data)

    if type is 'avg':
        return out
    elif type is 'var':
        return out_var


def loadWorker(f_location):
    data = []
    for t in tol_list:
        data.append(import_data(t, f_location))

    if not all(x is None for x in data):  # if all values aren't None
        data_filter = [x for x in data if x] # filter out None values
        data_avg = average_data(data_filter, type='avg', verbose=True)
        # data_var = average_data(data_filter, type='var')

        k = (pd.DataFrame.from_dict(data=data_avg, orient='columns'))
        k.columns = ['C/O ratio', 'SynGasSelec', 'SynGasYield', 'COSelec', 'COYield','H2Selec',
                      'H2Yield', 'CH4Conv', 'FullOxSelec', 'FullOxYield', 'ExitT',
                      'MaxT', 'DistToMaxT', 'O2Conv']
        # k.columns = ['C/O ratio', 'CH4 in', 'CH4 out', 'CO out', 'H2 out', 'H2O out', 'CO2 out', 'Exit temp', 'Max temp', 'Dist to max temp', 'O2 conv', 'Max CH4 Conv', 'Dist to 50 CH4 Conv']
        k.to_csv(f_location + '/avgdata.csv', header=True)

        # k = (pd.DataFrame.from_dict(data=data_var, orient='columns'))
        # k.columns = ['C/O ratio', 'CH4 in', 'CH4 out', 'CO out', 'H2 out', 'H2O out', 'CO2 out', 'Exit temp', 'Max temp', 'Dist to max temp', 'O2 conv', 'Max CH4 Conv', 'Dist to 50 CH4 Conv']
        # k.to_csv(f_location + '/vardata.csv', header=True)


def import_sensitivities(input, file_location):
    """
    Ratio is the C/O starting gas ratio
    file_location is the LSR C and O binding energy, false to load the base case
    """

    tol, ratio = input

    try:
        data = pd.read_csv(file_location + '/all-sensitivities/' + tol + '{:.1f}RxnSensitivity.csv'.format(ratio))

        data = data.values
        data = data.tolist()
        return data
    except:
        print('Cannot find ' + file_location + '/all-sensitivities/' + tol + '{:.1f}RxnSensitivity.csv'.format(ratio))


def average_sensitivities(data, type='avg'):
    """
    After loading in all the data at different ratios, average them all together
    to calculate one "master" sensitivity value
    """

    tol = len(data)
    rxn = len(data[0])
    sens = len(data[0][0])


    out = []
    out_var = []
    for r in range(rxn):
        fixed_data = []
        var_data = []
        for s in range(1, sens):
            tmp_list = []
            for t in range(tol):
                tmp_list.append(data[t][r][s])

            if s is 1:
                fixed_data.append(data[t][r][s])
                var_data.append(data[t][r][s])
            else:
                new_data = filter_outliers(tmp_list)
                avg = statistics.mean(new_data)
                var = statistics.variance(new_data)

                fixed_data.append(avg)
                var_data.append(var)

        out.append(fixed_data)
        out_var.append(var_data)

    if type is 'avg':
        return out
    elif type is 'var':
        return out_var


def loadSensDataWorker(f_location):
    for ratio in ratios:
        allsens = []
        for t in tol_list:
            i = (t, ratio)
            allsens.append(import_sensitivities(i, f_location))

        if not all(x is None for x in allsens):  # if all values aren't None
            os.path.exists(f_location + '/avg-sensitivities/') or os.makedirs(f_location + '/avg-sensitivities/')
            allsens_filter = [x for x in allsens if x] # filter out None values
            sensdata = average_sensitivities(allsens_filter, type='avg')
            sens_var = append(average_sensitivities(allsens_filter, type='var'))

            k = (pd.DataFrame.from_dict(data=sensdata, orient='columns'))
            k.columns = ['Reaction', 'SYNGAS Selec', 'SYNGAS Yield', 'CO Selectivity', 'CO % Yield', 'H2 Selectivity', 'H2 % Yield',
                         'CH4 Conversion', 'H2O+CO2 Selectivity', 'H2O+CO2 yield', 'Exit Temp', 'Peak Temp',
                         'Dist to peak temp', 'O2 Conversion', 'Max CH4 Conv', 'Dist to 50 CH4 Conv']
            k.to_csv(f_location + '/avg-sensitivities/{:.1f}avgRxnSensitivity.csv'.format(ratio), header=True)  # raw data

            k = (pd.DataFrame.from_dict(data=sens_var, orient='columns'))
            k.columns = ['Reaction', 'SYNGAS Selec', 'SYNGAS Yield', 'CO Selectivity', 'CO % Yield', 'H2 Selectivity', 'H2 % Yield',
                         'CH4 Conversion', 'H2O+CO2 Selectivity', 'H2O+CO2 yield', 'Exit Temp', 'Peak Temp',
                         'Dist to peak temp', 'O2 Conversion', 'Max CH4 Conv', 'Dist to 50 CH4 Conv']
            k.to_csv(f_location + '/avg-sensitivities/{:.1f}varRxnSensitivity.csv'.format(ratio), header=True)  # raw data


num_threads = len(ratios)
lump = 16
pool = multiprocessing.Pool(processes=num_threads)
all_data = pool.imap(loadWorker, allfiles, lump)
pool.close()
pool.join()

num_threads = max_cpus
pool = multiprocessing.Pool(processes=num_threads)
sens_data = pool.imap(loadSensDataWorker, allfiles, lump)
pool.close()
pool.join()
