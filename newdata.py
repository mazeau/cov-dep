import os
import re
import pandas as pd
import sys
import multiprocessing
import itertools
import statistics
import numpy as np


max_cpus = multiprocessing.cpu_count()
array = os.listdir('linearscaling/')
array = sorted(array)
ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
tol_list = ['10_16_', '10_17_', '10_18_', '10_19_', '10_20_', '10_21_',
            '10_22_', '10_23_', '10_24_', '11_16_', '11_17_', '11_18_',
            '11_19_', '11_20_', '11_21_', '11_22_', '11_23_', '11_24_',
            '12_16_', '12_17_', '12_18_', '12_19_', '12_20_', '12_21_',
            '12_22_', '12_23_', '12_24_',]
# get all combinations of the above for importing
inputs = list(itertools.product(tol_list, ratios))


def import_data(tol, file_location):
    try:
        data = pd.read_csv('newdata/' + file_location + '/' + tol + 'data.csv')

        data = data.values
        data = data.tolist()
        return data
    except:
        print('Cannot find ' + file_location + '/' + tol + 'data.csv')


def calculate(data):
    # with the import data, calculate the things we actually are interested in
    ratio = data[1]
    ch4_in = data[2]
    ch4_out = data[3]
    co_out = data[4]
    h2_out = data[5]
    h2o_out = data[6]
    co2_out = data[7]
    exit_T = data[8]
    max_T = data[9]
    dist_Tmax = data[10]
    o2_conv = data[11]
    max_ch4_conv = data[12]  # max rate of change
    dist_to_50_ch4_conv = data[13]

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

    if ch4_conv < 0.5:
        dist_to_50_ch4_conv = 110.

    return syngas_sel, syngas_yield, co_sel, co_yield, h2_sel, h2_yield, ch4_conv, fullox_sel, fullox_yield, exit_T, max_T, dist_Tmax, o2_conv, max_ch4_conv, dist_to_50_ch4_conv


def average_data(data, type='avg'):
    tol = len(data)
    ratio = len(data[0])
    value = len(data[0][0])

    out = []
    out_var = []
    for r in range(ratio):
        fixed_data = []
        var_data = []
        for s in range(2, value):
            tmp_list = []
            for t in range(tol):
                tmp_list.append(data[t][r][s])

            avg = statistics.mean(tmp_list)
            var = statistics.variance(tmp_list)
            fixed_data.append(avg)
            var_data.append(var)
        fixed_data.insert(0, data[0][r][1])
        fixed_data.insert(0, data[0][r][0])
        var_data.insert(0, data[0][r][1])
        var_data.insert(0, data[0][r][0])
        out.append(calculate(fixed_data))
        out_var.append(var_data)

    if type is 'avg':
        return out
    elif type is 'var':
        return out_var


def loadWorker(f_location):
    data = []
    for t in tol_list:
        data.append(import_data(t, f_location))

    data_filter = [x for x in data if x] # filter out None values
    data_avg = average_data(data_filter, type='avg')
    data_var = average_data(data_filter, type='var')

    return data_avg


# def import_data(input, file_location):
#     """
#     This imports data.csv from the original simulation
#     input is a tuple of the tolerances and ratio
#     """
#     data = pd.read_csv('newdata/' + file_location + '/' + tol + 'data.csv')
#
#     data = data.values
#     for x in range(len(data)):
#         r = round(data[x][1],1)
#         if r == ratio:
#             return calculate(data[x])
#
#
# def loadWorker(input):
#     ans = []
#     for f in array:
#         for t in tol_list:
#             ans.append(import_data(ratio, t, file_location=f))
#     return ans


def import_sensitivities(input, file_location):
    """
    Ratio is the C/O starting gas ratio
    file_location is the LSR C and O binding energy, false to load the base case
    """

    tol, ratio = input

    try:
        data = pd.read_csv('newdata/' + file_location + '/' + tol + str(ratio) + 'RxnSensitivity.csv')

        data = data.values
        data = data.tolist()
        return data
    except:
        print('Cannot find ' + file_location + '/' + tol + str(ratio) + 'RxnSensitivity.csv')


def average_sensitivities(data, type='avg'):
    """
    After loading in all the data at different ratios, average them all together
    to calculate one "master" sensitivity value

    Yes, it does both but only returns the one that is called.
    Will rewrite to make it more efficient at a later point in time.
    """

    tol = len(data)
    rxn = len(data[0])
    sens = len(data[0][0])


    out = []
    out_var = []
    for r in range(rxn):
        fixed_data = []
        var_data = []
        for s in range(2, sens-1):
            tmp_list = []
            for t in range(tol):
                tmp_list.append(data[t][r][s])

            avg = statistics.mean(tmp_list)
            var = statistics.variance(tmp_list)
            data_new = np.array(tmp_list)
            q25, q75 = np.percentile(data_new, 25), np.percentile(data_new, 75)
            iqr = q75 - q25
            cut_off = iqr * 2
            lower, upper = q25 - cut_off, q75 + cut_off
            outliers = [x for x in data_new if x < lower or x > upper]
            outliers_removed = [x for x in data_new if x >= lower and x <= upper]
            print(f"Removing {len(outliers)} outliers")
            avg = statistics.mean(outliers_removed)
            var = statistics.variance(outliers_removed)

            fixed_data.append(avg)
            var_data.append(var)
        fixed_data.insert(0, data[0][r][1])
        # fixed_data.insert(0, data[0][r][0])
        var_data.insert(0, data[0][r][1])
        # var_data.insert(0, data[0][r][0])
        out.append(fixed_data)
        out_var.append(var_data)

    if type is 'avg':
        return out
    elif type is 'var':
        return out_var


def loadSensDataWorker(f_location):
    sensdata = []
    sens_var = []

    for ratio in ratios:
        allsens = []
        for t in tol_list:
            i = (t, ratio)
            allsens.append(import_sensitivities(i, f_location))

        allsens_filter = [x for x in allsens if x] # filter out None values
        sensdata.append(average_sensitivities(allsens_filter, type='avg'))
        sens_var.append(average_sensitivities(allsens_filter, type='var'))

    return sensdata


num_threads = len(ratios)
pool = multiprocessing.Pool(processes=num_threads)
all_data = pool.map(loadWorker, array, 1)
pool.close()
pool.join()

print(all_data)

num_threads = max_cpus
lump = 1
pool = multiprocessing.Pool(processes=num_threads)
sens_data = pool.map(loadSensDataWorker, array, lump)
pool.close()
pool.join()
