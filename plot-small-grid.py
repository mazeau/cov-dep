#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})

import os
import re
import pandas as pd
import numpy as np
import shutil
import subprocess
import multiprocessing
import re
import cantera as ct
from matplotlib import animation
import sys
import statistics
import itertools

max_cpus = multiprocessing.cpu_count()

# set up the LSR grid, for the smaller, more interesting one
carbon_range = (-7.5, -5.5)
oxygen_range = (-5.25, -3.25)
grid_size = 9
mesh  = np.mgrid[carbon_range[0]:carbon_range[1]:grid_size*1j,
                 oxygen_range[0]:oxygen_range[1]:grid_size*1j]

with sns.axes_style("whitegrid"):
    plt.axis('square')
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3,0.5))
# plt.show()

# just to double-check
experiments = mesh.reshape((2,-1)).T

with sns.axes_style("whitegrid"):
    plt.axis('square')
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3,0.5))
    plt.plot(*experiments.T, marker='o', linestyle='none')
plt.clf()
extent = carbon_range + oxygen_range

# Because the center of a corner pixel is in fact the corner of the grid
# Becaus we want to stretch the image a little
c_step = mesh[0,1,0]-mesh[0,0,0]
o_step = mesh[1,0,1]-mesh[1,0,0]
carbon_range2 = (carbon_range[0]-c_step/2, carbon_range[1]+c_step/2)
oxygen_range2 = (oxygen_range[0]-c_step/2, oxygen_range[1]+c_step/2)
extent2 = carbon_range2 + oxygen_range2

# For close packed surfaces from
# Abild-Pedersen, F.; Greeley, J.; Studt, F.; Rossmeisl, J.; Munter, T. R.;
# Moses, P. G.; Skúlason, E.; Bligaard, T.; Norskov, J. K.
# Scaling Properties of Adsorption Energies for Hydrogen-Containing Molecules on
# Transition-Metal Surfaces. Phys. Rev. Lett. 2007, 99 (1), 016105
# DOI: 10.1103/PhysRevLett.99.016105.
abildpedersen_energies = { # Carbon, then Oxygen
    'Pt':(-6.363636363636363,-3.481481481481482),
    'Rh':(-6.5681818181818175,-4.609771721406942),
    # 'Ir':(-6.613636363636363,-5.94916142557652),
    # 'Au':(-3.7499999999999973,-2.302236198462614),
    'Pd':(-6, -3.517877940833916),
    # 'Cu':(-4.159090909090907,-3.85272536687631),
    # 'Ag':(-2.9545454545454533,-2.9282552993244817),
    'Ni':(-6.045454545454545,-4.711681807593758),
    'Ru':(-6.397727272727272,-5.104763568600047),
}

# "A Framework for Scalable Adsorbate-adsorbate Interaction Models"
# Max J Hoffmann, Andrew J Medford, and Thomas Bligaard
# From 2016 Hoffman et al.
# https://doi.org/10.1021/acs.jpcc.6b03375
hoffman_energies = { # Carbon, then Oxygen
    'Pt':(-6.750,-3.586),
    'Rh':(-6.78,-5.02),
    'Ir':(-6.65,-4.73),
    'Pd':(-6.58,-4.38),
    # 'Cu':(-4.28,-4.51),
    # 'Ag':(-2.91,-3.55),
    'Ni':(-6.045454545454545,-4.711681807593758),# copied from abild pedersen
    'Ru':(-6.397727272727272,-5.104763568600047),# copied from abild pedersen
}


def calculate(data):
    index, ratio, ch4_in, ch4_out, co_out, h2_out, h2o_out, co2_out, exit_T, max_T, dist_Tmax, o2_conv, max_ch4_conv, dist_50_ch4_conv = data

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

    return syngas_sel, syngas_yield, co_sel, co_yield, h2_sel, h2_yield, ch4_conv, fullox_sel, fullox_yield, exit_T, max_T, dist_Tmax, o2_conv


def import_data(ratio, f_location):
    """
    This imports data from the original simulation
    """
    data = pd.read_csv(str(f_location) + '/avgdata.csv')

    data = data.values
    data = data.tolist()

    for x in data:
        if x[1] == ratio:
            calculated_data = calculate(x)
            return list(calculated_data)


def lavaPlot(overall_rate, title, axis=False, folder=False, interpolation=True):
    """
    Overall data to plot in a 9x9 LSR grid

    Title is a string for what definition is used
    Axis is a list of a min and max value or False.  This is to normalize colors across many plots
    Folder is a string that specifies where to save the images
    Interpolation is False to just plot boxes
    """
    overall_rate = np.array(overall_rate)
    rates = overall_rate

    rates_grid = np.reshape(rates, (grid_size,grid_size))
    for i in range(0,8):
        for j in range(0, 8 - i):
            rates_grid[i][j], rates_grid[8 - j][8 - i] = rates_grid[8 - j][8 - i], rates_grid[i][j]
    if axis is False:  # no normalizing
        if interpolation is True:
            plt.imshow(rates_grid, origin='lower',
                       interpolation='spline16',
                       extent=extent2, aspect='equal', cmap="Spectral_r",)
        else:
            plt.imshow(rates_grid, origin='lower',
                       extent=extent2, aspect='equal', cmap="Spectral_r",)
    else:
        if interpolation is True:
            plt.imshow(rates_grid, origin='lower',
                       interpolation='spline16',
                       extent=extent2, aspect='equal', cmap="Spectral_r",
                       vmin=axis[0], vmax=axis[1],)
        else:
            plt.imshow(rates_grid, origin='lower',
                       extent=extent2, aspect='equal', cmap="Spectral_r",
                       vmin=axis[0], vmax=axis[1],)

    for metal, coords in hoffman_energies.items():
        # color = {'Ag':'k','Au':'k','Cu':'k'}.get(metal,'k')
        color = {'Pt':'k','Rh':'k','Ir':'k','Pd':'k','Ni':'k','Ru':'k'}.get(metal,'k')
        plt.plot(coords[0], coords[1], 'o'+color)
        plt.text(coords[0], coords[1]-0.1, metal, color=color, fontsize=16)
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3.,0.5))
    plt.xlabel('$\Delta E^C$ (eV)', fontsize=22)
    plt.ylabel('$\Delta E^O$ (eV)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.colorbar().ax.tick_params(labelsize=18)
    out_dir = 'lsr'
    os.path.exists(out_dir) or os.makedirs(out_dir)
    if folder is False:
        plt.savefig(out_dir + '/' + str(title) +'.pdf', bbox_inches='tight')
    else:
        os.path.exists(out_dir + '/' + str(folder)) or os.makedirs(out_dir + '/' + str(folder))
        plt.savefig(out_dir + '/' + str(folder) + '/' + str(title) +'.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()


def lavaPlotAnimate(overall_rate, title, axis=False, folder=False, interpolation=True):
    """
    Overall data to plot in a 9x9 LSR grid

    Title is a string for what definition is used
    Axis is a list of a min and max value or False.  This is to normalize colors across many plots
    Folder is a string that specifies where to save the images
    Interpolation is False to just plot boxes
    """
    fig = plt.figure()
    ims = []

    for ratio in range(len(overall_rate)):
        rates = np.array(overall_rate[ratio])

        rates_grid = np.reshape(rates, (grid_size,grid_size))
        for i in range(0,8):
            for j in range(0, 8 - i):
                rates_grid[i][j], rates_grid[8 - j][8 - i] = rates_grid[8 - j][8 - i], rates_grid[i][j]
        if axis is False:  # no normalizing
            if interpolation is True:
                im = plt.imshow(rates_grid, origin='lower',
                           interpolation='spline16',
                           extent=extent2, aspect='equal', cmap="Spectral_r",
                           animated=True)
            else:
                im = plt.imshow(rates_grid, origin='lower',
                           extent=extent2, aspect='equal', cmap="Spectral_r",
                           animated=True)
        else:
            if interpolation is True:
                im = plt.imshow(rates_grid, origin='lower',
                           interpolation='spline16',
                           extent=extent2, aspect='equal', cmap="Spectral_r",
                           vmin=axis[0], vmax=axis[1], animated=True)
            else:
                im = plt.imshow(rates_grid, origin='lower',
                           extent=extent2, aspect='equal', cmap="Spectral_r",
                           vmin=axis[0], vmax=axis[1], animated=True)
        ims.append([im])
    for metal, coords in hoffman_energies.items():
        # color = {'Ag':'k','Au':'k','Cu':'k'}.get(metal,'k')
        color = {'Pt':'k','Rh':'k','Ir':'k','Pd':'k','Ni':'k','Ru':'k'}.get(metal,'k')
        plt.plot(coords[0], coords[1], 'o'+color)
        plt.text(coords[0], coords[1]-0.1, metal, color=color, fontsize=16)
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3.,0.5))
    plt.xlabel('$\Delta E^C$ (eV)', fontsize=22)
    plt.ylabel('$\Delta E^O$ (eV)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.colorbar().ax.tick_params(labelsize=18)
    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=300, blit=True)
    out_dir = 'lsr'
    os.path.exists(out_dir) or os.makedirs(out_dir)
    if folder is False:
        ani.save(out_dir + '/' + str(title) + '.gif', writer='pillow', fps=5)
    else:
        os.path.exists(out_dir + '/' + str(folder)) or os.makedirs(out_dir + '/' + str(folder))
        ani.save(out_dir + '/' + str(folder) + '/' + str(title) + '.gif', writer='pillow', fps=5)
        # ani.save(out_dir + '/' + str(folder) + '/' + str(title) + '.mpg', writer='ffmpeg', fps=5)


array = os.listdir('small-grid/')
array = sorted(array)
files = ['./small-grid/' + x for x in array]
# array = os.listdir('small-grid-cov/')
# array = sorted(array)
# allfiles = files + ['./small-grid-cov/' + x for x in array] + ['./pt_cathub/','./pt_cathub_cov/']


# for plotting
c_s = []
o_s = []
for x in array:
    _, c, o = x.split("-")
    c = c[:-1]
    c = -1 *float(c)
    o = -1* float(o)
    c_s.append(c)
    o_s.append(o)

ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
ratios_title = ['06', '07', '08', '09', '10', '11', '12', '13', '14', '16', '18', '20', '22', '24', '26']
sens_types = ['SynGasSelec', 'SynGasYield', 'COSelec', 'COYield', 'H2Selec',
              'H2Yield', 'CH4Conv', 'FullOxSelec', 'FullOxYield', 'ExitT',
              'MaxT', 'DistToMaxT', 'O2Conv']


def loadWorker(ratio):
    data = []

    array = os.listdir('small-grid/')
    array = sorted(array)
    files = ['./small-grid/' + x for x in array]

    for f in files:
        data.append(import_data(ratio, f))

    return data


num_threads = len(ratios)
pool = multiprocessing.Pool(processes=num_threads)
all_data = pool.map(loadWorker, ratios, 1)
pool.close()
pool.join()

def spansWorker(sens):
    all_sens_data = []
    for x in range(len(all_data)):  # for each ratio
        for y in range(len(all_data[0])):
            # x has len 15 and is each of the ratios
            # y has len 81 and is each of the lsr binding energies
            # the last number is the type of sensitivity definition and is 0-12
            all_sens_data.append(all_data[x][y][sens])
    vmax = max(all_sens_data)
    vmin = min(all_sens_data)
    return [vmin, vmax]


def basePlotWorker(ratio):
    # plots without interpolation
    for s in range(len(all_data[ratio][0])):
        data_to_plot = []
        for x in range(len(all_data[ratio])):
            data_to_plot.append(all_data[ratio][x][s])
        title = sens_types[s] + str(ratios_title[ratio])
        lavaPlot(data_to_plot, title, axis=spans[s], folder='base', interpolation=False)


def baseAnimateWorker(sens):
    # make gifs
    data_to_plot = []
    for ratio in range(len(all_data)):
        tmp = []
        for metal in range(len(all_data[0])):
            tmp.append(all_data[ratio][metal][sens])
        data_to_plot.append(tmp)
    title = sens_types[sens]
    lavaPlotAnimate(data_to_plot, title, spans[sens], folder='base-animate', interpolation=False)


sens_index = list(range(len(sens_types)))
pool = multiprocessing.Pool(processes=15)
spans = pool.map(spansWorker, sens_index, 1)
pool.close()
pool.join()

ratios_index = list(range(len(ratios)))

if max_cpus >= 28:
    num_threads = 28
    lump = 1
else:
    num_threads = max_cpus
    lump = int(28./max_cpus)
pool = multiprocessing.Pool(processes=num_threads)
pool.map_async(basePlotWorker, ratios_index)
pool.map_async(baseAnimateWorker, sens_index)
pool.close()
pool.join()

# pool = multiprocessing.Pool(processes=15)
# t = pool.map(basePlotWorker, ratios_index, 1)
# pool.close()
# pool.join()
#
# pool = multiprocessing.Pool(processes=15)
# pool.map(baseAnimateWorker, sens_index, 1)
# pool.close()
# pool.join()


def import_sensitivities(ratio, file_location, avg='avg'):
    """
    Ratio is the C/O starting gas ratio
    file_location is the LSR C and O binding energy, false to load the base case
    """

    try:
        data = pd.read_csv(file_location + '/avg-sensitivities/' + str(ratio) + avg + 'RxnSensitivity.csv')

        data = data.values
        data = data.tolist()
        return data
    except:
        print('Cannot find ' + file_location + '/avg-sensitivities/' + str(ratio) + avg + 'RxnSensitivity.csv')


def loadSensDataWorker(f_location):
    sensdata = []
    sens_var = []

    for ratio in ratios:
        sensdata.append(import_sensitivities(ratio, f_location))
        sens_var.append(import_sensitivities(ratio, f_location, avg='var'))

    return sensdata


num_threads = max_cpus
lump = 1
pool = multiprocessing.Pool(processes=num_threads)
allrxndata = pool.map(loadSensDataWorker, files, lump)
pool.close()
pool.join()

reactions = set()  # create list of unique reactions
for f in range(len(allrxndata)):  # for each lsr binding energy
    for r in range(len(allrxndata[f][6])):  # for each reaction
        reactions.add(allrxndata[f][6][r][1])  # append the reaction itself
reactions = list(reactions)

sys.exit()

def sensPlot(overall_rate, title, axis=False, folder=False):
    """
    overall sensitivity data to plot
    title is a string for what definition is used
    to normalize colors across many plots, False doesn't normalize axes
    folder specifies where to save the images
    """
    cmap = plt.get_cmap("Spectral_r")
    # cmap.set_bad(color='k', alpha=None)
    cmaplist = list(map(cmap,range(256)))
    cmaplist[0]=(0,0,0,0.3)
    newcmap = cmap.from_list('newcmap',cmaplist, N=256)
    cmap = newcmap

    overall_rate = np.array(overall_rate)
    rates = overall_rate

    rates_grid = np.reshape(rates, (grid_size,grid_size))
    for i in range(0,8):
        for j in range(0, 8 - i):
            rates_grid[i][j], rates_grid[8 - j][8 - i] = rates_grid[8 - j][8 - i], rates_grid[i][j]
    if axis is False:  # no normalizing
        plt.imshow(rates_grid, origin='lower',
                   extent=extent2, aspect='equal', cmap="Spectral_r",)
    else:
        plt.imshow(rates_grid, origin='lower',
               extent=extent2, aspect='equal', cmap="Spectral_r",
               vmin=axis[0], vmax=axis[1],)

    for metal, coords in abildpedersen_energies.items():
        color = {'Ag':'k','Au':'k','Cu':'k'}.get(metal,'k')
        plt.plot(coords[0], coords[1], 'o'+color)
        plt.text(coords[0], coords[1]-0.1, metal, color=color, fontsize=16)
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3.,0.5))
    plt.xlabel('$\Delta E^C$ (eV)', fontsize=22)
    plt.ylabel('$\Delta E^O$ (eV)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.colorbar().ax.tick_params(labelsize=18)
    out_dir = 'lsr'
    os.path.exists(out_dir) or os.makedirs(out_dir)
    if folder is False:
        plt.savefig(out_dir + '/' + str(title) +'.pdf', bbox_inches='tight')
    else:
        os.path.exists(out_dir + '/' + str(folder)) or os.makedirs(out_dir + '/' + str(folder))
        plt.savefig(out_dir + '/' + str(folder) + '/' + str(title) +'.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()


def sensPlotWorker(input):
    rxn, s = input
    tot_sens = 0.

    for r in range(len(allrxndata[0])):  # for a single ratio
        sensitivities = []
        for f in range(len(array)):  # for lsr binding energies
            got_value = False
            for p in range(len(allrxndata[f][r])):  # matching the reaction
                if allrxndata[f][r][p][1] == np.str(rxn):
                    sensitivities.append(allrxndata[f][r][p][s+2])
                    got_value = True
            if got_value is False:
                # this reaction didn't show up on this metal, so it isn't
                # sensitive, so put a placeholder in
                sensitivities.append(0.)

        tot_sens += sum(abs(np.array(sensitivities)))
        STDEV = statistics.stdev(sensitivities)
        MAX = max(abs(np.array(sensitivities)))

        title = rxn + ' '+ sens_types[s-2] + ' ' + str(ratios[r])
        sensPlot(sensitivities, title, folder='rxnsensitivities', axis=[-1*MAX, MAX])
        sensPlot(sensitivities, title, folder='rxnsensitivities-stdev', axis=[-1*STDEV*2, STDEV*2])

    return [rxn, tot_sens, sens_types[s]]


def sensPlotAnimate(overall_rate, title, axis=False, folder=False):
    """
    overall sensitivity data to plot
    title is a string for what definition is used
    to normalize colors across many plots, False doesn't normalize axes
    folder specifies where to save the images
    """
    fig = plt.figure()
    ims = []

    cmap = plt.get_cmap("Spectral_r")
    # cmap.set_bad(color='k', alpha=None)
    cmaplist = list(map(cmap,range(256)))
    cmaplist[0]=(0,0,0,0.3)
    newcmap = cmap.from_list('newcmap',cmaplist, N=256)
    cmap = newcmap

    for ratio in range(len(overall_rate)):
        rates = np.array(overall_rate[ratio])

        rates_grid = np.reshape(rates, (grid_size,grid_size))
        for i in range(0,8):
            for j in range(0, 8 - i):
                rates_grid[i][j], rates_grid[8 - j][8 - i] = rates_grid[8 - j][8 - i], rates_grid[i][j]
        if axis is False:  # no normalizing
            im = plt.imshow(rates_grid, origin='lower',
                       extent=extent2, aspect='equal', cmap="Spectral_r",)
        else:
            im = plt.imshow(rates_grid, origin='lower',
                   extent=extent2, aspect='equal', cmap="Spectral_r",
                   vmin=axis[0], vmax=axis[1],)
        ims.append([im])

    for metal, coords in abildpedersen_energies.items():
        color = {'Ag':'k','Au':'k','Cu':'k'}.get(metal,'k')
        plt.plot(coords[0], coords[1], 'o'+color)
        plt.text(coords[0], coords[1]-0.1, metal, color=color, fontsize=16)
    plt.xlim(carbon_range)
    plt.ylim(oxygen_range)
    plt.yticks(np.arange(-5.25,-3,0.5))
    plt.xlabel('$\Delta E^C$ (eV)', fontsize=22)
    plt.ylabel('$\Delta E^O$ (eV)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.colorbar().ax.tick_params(labelsize=18)
    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=300, blit=True)
    out_dir = 'lsr'
    os.path.exists(out_dir) or os.makedirs(out_dir)
    if folder is False:
        ani.save(out_dir + '/' + str(title) + '.gif', writer='animation.PillowWriter', fps=5)
    else:
        os.path.exists(out_dir + '/' + str(folder)) or os.makedirs(out_dir + '/' + str(folder))
        ani.save(out_dir + '/' + str(folder) + '/' + str(title) + '.gif', writer='pillow', fps=5)
        # ani.save(out_dir + '/' + str(folder) + '/' + str(title) + '.mpg', writer='ffmpeg', fps=5)
    plt.clf()


def sensPlotAnimateWorker(input):
    rxn, s = input
    print("{}".format(s))

    sensitivities = []
    for r in range(len(allrxndata[0])):  # for a single ratio
        tmp_sens = []
        for f in range(len(array)):  # for lsr binding energies
            got_value = False
            for p in range(len(allrxndata[f][r])):  # matching the reaction
                if allrxndata[f][r][p][1] == np.str(rxn):
                    tmp_sens.append(allrxndata[f][r][p][s+2])
                    got_value = True
            if got_value is False:
                # this reaction didn't show up on this metal, so it isn't
                # sensitive, so put a placeholder in
                tmp_sens.append(0.)
        if len(tmp_sens) != 81:
            print("Skipping {} {} because sensitivity len is {} but should be 81".format(rxn, sens_index[s], len(tmp_sens)))
            continue
        else:
            sensitivities.append(tmp_sens)
    # standardizing the colors across all ratios
    flat = [item for sublist in sensitivities for item in sublist]
    MAX = max(abs(np.array(flat)))
    STDEV = statistics.stdev(flat)
    # AVG = (sum(abs(np.array(flat)))/len(flat))*1.5  # cutoff the color plot at x times the average sensitivity
    title = str(rxn) + str(sens_types[s])
    sensPlotAnimate(sensitivities, title, axis=[-1*MAX,MAX], folder='rxnsensitivities-animate')
    sensPlotAnimate(sensitivities, title, axis=[-1*STDEV*2,STDEV*2], folder='rxnsensitivities-animate-stdev')

    return [rxn, sens_types[s-2], MAX]


num_threads = max_cpus
lump = int(len(reactions)*15/max_cpus)+1
input = list(itertools.product(reactions, sens_index))
pool = multiprocessing.Pool(processes=num_threads)
sum_sens = pool.map(sensPlotWorker, input, lump)
pool.close()
pool.join()

pool = multiprocessing.Pool(processes=num_threads)
max_sens = pool.map(sensPlotAnimateWorker, input, lump)
pool.close()
pool.join()

sorted_max_sens = sorted(max_sens, key=lambda l:l[2], reverse=True)
for x in sorted_max_sens:
    print('{}\t{}\t{}'.format(x[0], x[1], x[2]))
k = pd.DataFrame.from_records(sorted_max_sens, columns=['Reaction', 'Sens Type', 'Maximum'])
k.to_csv('maxsens.csv', header=True)

for x in sum_sens:
    print('{}\t{}\t{}'.format(x[0], x[1], x[2]))
k = pd.DataFrame.from_records(sum_sens, columns=['Reaction', 'Total Sens', 'Sens Type'])
k.to_csv('sumsens.csv', header=True)
