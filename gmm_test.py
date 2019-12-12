"""
Author: Ramsin Khoshabeh
Contact: ramsin@ucsd.edu
Date: 07 April 2019

Description: An entry point 
"""

import traceback
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import normalize
from itertools import islice 

from mywearable.ble import BLE
from mywearable.ppg import PPG

show_plots=True

""" -------------------- Settings -------------------- """
run_config = False                # whether to config PC HM-10 or not
baudrate = 9600                   # PySerial baud rate of the PC HM-10
serial_port = 'COM28'              # Serial port of the PC HM-10
peripheral_mac = '78DB2F13E9D2'   # Mac Address of the Arduino HM-10
# 78DB2F168240

""" -------------------- File Code -------------------- """
def check_fs(time_data, sampling_rate):
    diffs = np.diff(time_data)
    avg_diff = np.mean(diffs)
    estimated_fs = 1e3/avg_diff # assumes time data is stored in milliseconds!
    if estimated_fs < sampling_rate:
        print('Warning: Low FS detected! Re-collect data with an FS less than %3.2f Hz.' % estimated_fs)
    return estimated_fs

def get_subjects(filenames):
    '''
    takes in a list of the filenames in a given directory and then extracts the name of each 
    unique subject and returns a list with those names. It assumes files are named using the 
    convention of Objective 1: GMM_<initials>_<trial>.csv
    Hint: you can use glob to retrieve a list of all the *.csv files in the data directory.
    '''
    subjects = []
    for filename in filenames:
        _, name = filename.rstrip('.csv').split('GMM_')
        subject, _ = name.split('_')
        if subject not in subjects:
            subjects.append(subject)
    return subjects

def load_files(subjects, trials, filenames, fs):
    '''
    takes in the list of subjects, list of trials, list of filenames, and the sampling rate (fs). 
    It loads all the data files for each subject/trial and appends them together into a large 
    list of time and PPG value arrays. You can use this method to load the data for your training, 
    validation, and testing sets separately.
    '''
    times = []
    values = []
    # Accumulate all data from 'trials' for every subject in 'subjects'
    for filename in filenames:
        _, name = filename.rstrip('.csv').split('GMM_')
        subject, trial = name.split('_')
        if subject in subjects and trial in trials:
            '''
            Since we're appending time data we should start the first value from     
            t = 0 and offset by the duration of the previous dataset when appending   
            the next one 
            '''
            try:
                time_prev = times[-1]
            except:
                time_prev = 0

            time_offset = 0
            value_offset = 0
            # Load the data from the text into two variables, t and ir. Remember your delimiter is ',' and we want integers as the data type
            # Call the check_fs function to check/confirm the sampling rate
            # Append the t and ir into the two arrays for times and values
            with open(filename, "r") as f:
                lines = f.readlines()
                t_list = []
                ir_list = []
                for index, line in enumerate(lines):
                    t, ir = line.split(',')
                    t, ir = int(t.strip()), int(ir.strip())
                    # print(t, ir)
                    t_list.append(t)
                    ir_list.append(ir)

                ir_list = sig.detrend(ir_list)  # demean
                ir_list = np.gradient(ir_list)  # gradient
                # ir_list_norm = [ (ir-min(ir_list))/(max(ir_list)-min(ir_list)) for ir in ir_list ]
                # ir_list_norm = normalize(ir_list[:,np.newaxis], axis=0).ravel() # fast normalize
                min_max_scaler = preprocessing.MinMaxScaler()
                ir_list_norm = min_max_scaler.fit_transform(ir_list[:,np.newaxis]).ravel()
                
                for t, ir in zip(t_list, ir_list_norm):
                    times.append( time_prev + t - t_list[0] )
                    values.append( ir )

        
    return times, values

def plot(x, y, overlay=None, title=None, filename=None, slice_window=None):
    '''
    Plots the data in the time and data buffers onto a figure
    :param: None
    :return: None
    '''
    plt.cla()
    if slice_window:
        plt.plot(x[slice_window], y[slice_window])
    else:
        plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('PPG Reading (Normalized)')
    if overlay:
        locations, values = zip(*overlay)
        plt.scatter(locations, values, color='red')
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    if show_plots:
        plt.show()
    return None

def hist(x, y, overlay=None, title=None, filename=None):
    plt.cla()
    
    histogram = plt.hist(y, bins=50)
    plt.xlabel('PPG Reading (Normalized)')
    plt.ylabel('Count (Normalized)')

    if overlay:
        locations, values = zip(*overlay)
        plt.scatter(locations, values, color='red')
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    if show_plots:
        plt.show()
    return histogram

def hist_gmm(data, gmm, plot_sum=True, overlay=None, title=None, filename=None):
    plt.cla()
    # if show_plots:
    #     plt.ion()

    mu1 = gmm.means_[0, 0]
    mu2 = gmm.means_[1, 0]
    var1, var2 = gmm.covariances_
    wgt1, wgt2 = gmm.weights_

    x = np.linspace(0, 1, 1000)

    plt.hist(data, bins=50, density=True, alpha=.3, label='Histogram')    
    plt.vlines((mu1, mu2), ymin=0, ymax=2.5, label='Fitted Means')

    norm1 = wgt1 * norm.pdf(x, mu1, np.sqrt(var1).reshape(1))
    norm2 = wgt2 * norm.pdf(x, mu2, np.sqrt(var2).reshape(1))

    if plot_sum:
        plt.plot(x, norm1+norm2, label='sum')        
    else:
        plt.plot(x, norm1, label='mu1')
        plt.plot(x, norm2, label='mu2')

    plt.legend(loc='best', fontsize='x-small')
    plt.xlabel('PPG Reading (Normalized)')
    plt.ylabel('Count (Normalized)')

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)

    if show_plots:
        plt.show()

    plt.cla()

    return gmm

def objective_2():
    # objective 2
    filenames = glob.glob('Lab 7/Objective 2/data/*.csv')
    subjects = get_subjects(filenames)

    trials = [ str(x) for x in range(1, 6) ]

    for subject in subjects:
        subjects_validate = subject
        subjects_train = [x for x in subjects if x!=subject]
        # print(subjects_validate, subjects_train)

        # 2
        gmm_validate_t, gmm_validate_ir = load_files(subjects_validate, trials, filenames, 25)
        gmm_train_t, gmm_train_ir = load_files(subjects_train, trials, filenames, 25)

        # 3
        title = 'Training with '+subjects_validate+' Left out'
        filename = 'training_data_'+subjects_validate+'.png'
        plot(gmm_train_t, gmm_train_ir, title=title, filename=filename, slice_window=None)

        # 4
        title = 'Histogram with '+subjects_validate+' Left out'
        filename = 'hist_training_'+subjects_validate+'.png'
        hist(gmm_train_t, gmm_train_ir, title=title, filename=filename)

        # 5
        data_train = np.array(gmm_train_ir).reshape(-1, 1)
        data_validate = np.array(gmm_validate_ir).reshape(-1, 1)

        # create gmm and fit
        gmm = GaussianMixture(n_components = 2)
        fit = gmm.fit(data_train)

        # sort the order of means
        # https://stackoverflow.com/questions/37008588/sklearn-gmm-classification-prediction-component-assignment-order
        sort_indices = gmm.means_.argsort(axis = 0)
        order = sort_indices[:, 0]
        gmm.means_ = gmm.means_[order,:]
        gmm.covariances_ = gmm.covariances_[order, :]
        w = np.split(gmm.weights_,2)
        w = np.asarray(w)
        w = np.ravel(w[order,:])
        gmm.weights_ = w

        title = 'IR Signal Histogram \n Individual with '+subjects_validate+' Left out'
        filename = 'hist_individual_'+subjects_validate+'.png'
        hist_gmm(data_train, gmm, plot_sum=False, title=title, filename=filename)

        title = 'IR Signal Histogram \n Sum with '+subjects_validate+' Left out'
        filename = 'hist_sum_'+subjects_validate+'.png'
        hist_gmm(data_train, gmm, plot_sum=True, title=title, filename=filename)

        #6
        predictions_train = gmm.predict(data_train)
        predictions_validate = gmm.predict(data_validate)

        # print(predictions_train)
        # print(predictions_validate)

        plt.ion()

        plt.cla()
        plt.plot(gmm_train_t[window], data_train[window], color='green', label='Voltage')
        plt.plot(gmm_train_t[window], predictions_train[window], color='red', label='Prediction')
        title = 'Prediction of Training Set \n with '+subjects_validate+' Left out'
        plt.title(title)
        plt.legend(loc='lower left', fontsize='small')
        plt.xlabel('Time')
        plt.ylabel('Reading')
        plt.pause(0.5)
        plt.show()
        filename = 'gmm_train_labeled_'+subjects_validate+'.png'
        plt.savefig(filename)

        plt.cla()
        plt.plot(gmm_validate_t[window], data_validate[window], color='green', label='Voltage')
        plt.plot(gmm_validate_t[window], predictions_validate[window], color='red', label='Prediction')
        title = 'Prediction of Validation Set \n with '+subjects_validate+' Left out'
        plt.title(title)
        plt.legend(loc='lower left', fontsize='small')
        plt.xlabel('Time')
        plt.ylabel('Reading')
        plt.pause(0.5)
        plt.show()
        filename = 'gmm_validate_labeled_'+subjects_validate+'.png'
        plt.savefig(filename)

def emergency_exit(event):
    if event.key == 'enter':
        quit()



if __name__ == "main":

    """ -------------------- Main Code -------------------- """
    show_plots=True
    window = slice(1100, 1300)   # arbitrary slice so we can see whats happening

    objective_2()

    # objective 3
    filenames = glob.glob('Lab 7/Objective 2/data/*.csv')
    filenames = np.random.permutation(filenames) # randomize
    num_files = len(filenames)
    trials = [ str(x) for x in range(1, 6) ]

    print('enter 0 to read from file or 1 for ble:')
    option = input()

    if option == '1':
        # objective 4
        seconds = 30
        ppg = PPG(seconds*25, 40)
        
        hm10 = BLE(serial_port, baudrate, run_config)
        # hm10.connect(peripheral_mac)

        # skip garbage
        for i in range(20):
            msg = hm10.read_line(eol=";", timeout=3)
        # actually read
        while True:
            msg = hm10.read_line(eol=";", timeout=3)
            print(msg)
            result = ppg.append(msg)
            if result == False:
                break


        subjects_all = get_subjects(filenames)
        t_training, ir_training = load_files(subjects_all, trials, filenames, 25)
        # print(ir_training)
        ppg.train(ir_training)

        ppg.filter_ppg()

        time = ppg.get_time()
        data = ppg.get_data()

        # print(time)
        # print(data)

        heart_rate = ppg.process(time, data)
        heart_rate_str = '%.3f'%heart_rate
        print('calculated heart rate:', heart_rate_str, 'bpm')

        hm10.write(heart_rate_str+';')


    else:
        # objective 3

        # partition data set
        files_iter = iter(filenames)
        files_training = list(islice(files_iter, int(0.6*num_files)))
        files_validate = list(islice(files_iter, int(0.3*num_files)))
        files_testing = list(islice(files_iter, int(0.1*num_files)))
        # debugging
        # print(len(files_training), files_training)
        # print(len(files_validate), files_validate)
        # print(len(files_testing), files_testing)

        subjects_training = get_subjects(files_training)
        subjects_validate = get_subjects(files_validate)
        subjects_testing = get_subjects(files_testing)

        t_training, ir_training = load_files(subjects_training, trials, filenames, 25)
        t_validate, ir_validate = load_files(subjects_validate, trials, filenames, 25)
        t_testing, ir_testing = load_files(subjects_testing, trials, filenames, 25)

        total_len = len(t_training) + len(t_validate) + len(t_testing)

        # train ppg with training set
        ppg = PPG(total_len, 40)
        ppg.train(ir_training)

        # save plots
        # ppg.plt_labels(ir_validate, t_validate, window=window, title='Validation Set', save='valid_labels.png')
        # ppg.plt_labels(ir_testing, t_testing, window=window, title='Testing Set', save='test_labels.png')

        # debugging
        # window_size = 300
        # for i in range(0, len(t_testing)-window_size, 50):
        #     window = slice(i, i+window_size)
        #     fig = plt.figure(1)
        #     fig.canvas.mpl_connect('key_press_event', emergency_exit)
        #     ppg.plt_labels(ir_testing, t_testing, window=window, title='Testing Set')
        #     plt.pause(0.01)

        # objective 4
        heart_rate = ppg.process(t_testing, ir_testing)
        print('calculated heart rate:', '%.3f'%heart_rate, 'bpm')

