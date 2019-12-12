"""
Authors: Ramsin Khoshabeh
Contact: ramsin@ucsd.edu
Date: 29 October 2019

Description: A class to handle the PPG for the wearable
"""

# Imports
import serial
from time import sleep
from time import time
from scipy import signal as sig
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import normalize
from itertools import islice 
import statistics as stat

class PPG:

    # Attributes of the class PPG
    _maxlen = 0
    __time_buffer = [] # private variable (encapsulation)
    __data_buffer = [] # private variable (encapsulation)
    __heartbeats = []
    __fs = 0 # sampling rate
    __lower_threshold = 0.45
    __higher_threshold = 100
    __model = None
    __heart_rate_list = list()
    __heart_rate = 0
    __counter = 0
    window_size = 0

    """ ================================================================================
    Constructor that sets up the PPG class. It will only run once.
    :param maxlen: (int) max length of the buffer
    :param fs: (int) sampling rate from the Arduino
    :return: None
    ================================================================================ """
    def __init__(self, maxlen, fs):
        self._maxlen = maxlen       # Set the max length of the buffer
        self.__fs = fs
        fig = plt.figure(1)
        fig.canvas.mpl_connect('key_press_event', self.__handle_keypress)
        return

    """ ================================================================================
    A callback function that triggers when a key is pressed with the plot open
    :param event: the input event that triggered the callback
    :return: None
    ================================================================================ """
    def __handle_keypress(self, event):
        if event.key == 'enter':
            self.save_file("PPG.csv")

    """ ================================================================================
    Resets the PPG to default state
    :return: None
    ================================================================================ """
    def reset(self):
        self.__time_buffer = []
        self.__data_buffer = []
        self.__heartbeats = []
        return

    """ ================================================================================
    Appends new elements to the data and time buffers by parsing 'msg_str' and splitting
    it, assuming comma separation. It also keeps track of buffer occupancy. Once the 
    buffer is full, it will become a circular buffer and drop samples from the beginning
    as a FIFO buffer.
    :param msg_str: (str) the string containing data that will be appended to the buffer
    :return: None
    ================================================================================ """
    def append(self, msg_str):

        ### TO DO ###
        # 1. Inside of a try-except block
            # 2.1. Split the incoming message by looking for ','
            # 2.2. Store the time to a temporary variable (make sure to cast as an int)
            # 2.3. Store the PPG data to temporary variable (make sure to cast as an int)
            #      - This will be on index 2 since IMU data is index 1!!!
        # 2. If there is a ValueError or IndexError exception
            # print that there was invalid data
        # 3. Check if our buffers are full by checking that the length of one of the 
        #    buffers (both should be have the same length) is equal to maxlen
            # 1.1. If the buffers are full, shift both buffers left by one value
            # 1.2. Assign the last value of the time buffer with the new time value
            # 1.3. Assign the last value of the data buffer with the new PPG data value
        # 4. If the buffers are not full, simply append the new time & data to the buffers
        # 5. Return nothing
        
        retval = True
        while len(self.__time_buffer) >= self._maxlen:
            self.__time_buffer.pop(0)
            self.__data_buffer.pop(0)
            retval = False
        try:
            result = [int(x.strip()) for x in msg_str.split(',')]
            self.__time_buffer.append(result[0])
            self.__data_buffer.append(result[2]) #index of ppg reading
            self.__counter += 1
        except ValueError:
            print("There wuz invalid data!")
        except IndexError:
            pass
        return retval

    """ ================================================================================
    Saves the contents of the buffer into the specified file one line at a time.
    :param filename: (str) the name of the file that will store the buffer data
    :return: None
    ================================================================================ """
    def save_file(self, filename):

        ### TO DO ###
        # 1. Open the file with mode set to write (this will overwrite any data)
        # 2. In a for-loop, iterate over the buffers
            # 2.1. Assemble a "row" string from the buffers formatted as "<t>,<d>\n"
            # 2.2. Write the row to the file
        # 3. Close the file or use the "with" keyword
        # 4. Return nothing

        # NOTE: This method is the same as in the Pedometer class
        f = open(filename, "w")
        combined_buffer = list(zip(self.__time_buffer, self.__data_buffer))
        for point in combined_buffer:
            row_str = "%d,%d\n" % (point[0],point[1])
            f.write(row_str)
        f.close()
        return None


    """ ================================================================================
    Loads the contents of the file 'filename' into the time and data buffers 
    :param filename: (str) the name (full path) of the file that we read from
    :return: None
    ================================================================================ """
    def load_file(self, filename):
        
        self.reset()

        ### TO DO ###
        # 1. Open the file with mode set to read
        # 2. Loop forever until _maxlen is reached
            # 2.1. Read a line from the file 
            # 2.2. Check to see if the line is None
                #2.2.1. If it is, break out of the loop (the end of file was reached)
            # 2.3. Strip any newline characters from the line and split it with ','
            # 2.4. Append time value to the time buffer
            # 2.5. Append data value to the data buffer
            # 2.6. Keep track of the number of times the while loop runs in a variable
        # 3. Return nothing

        # NOTE: This method is the same as in the Pedometer class
        f = open(filename, "r")
        count = 0
        while True:
            line_str = f.readline()
            if line_str==None or line_str=='':
                break
            values = [int(x.strip()) for x in line_str.split(',')]
            self.__time_buffer.append(values[0])
            self.__data_buffer.append(values[1])
            count += 1
        self._maxlen = count
        return None


    """ ================================================================================
    Plots the data in the time and data buffers onto a figure
    :param: None
    :return: None
    ================================================================================ """
    def plot(self, overlay=None, title=None, save_name=None):
        ## TO DO ##
        # 1. Open a new figure (if needed)
        # 2. Plot the time buffer versus the data buffer (stylize however you like)
        # 3. Show the plot
        # 4. Return nothing

        # NOTE: This method is the same as in the Pedometer class
        plt.cla()
        plt.plot(self.__time_buffer, self.__data_buffer)
        if overlay:
            locations, values = zip(*overlay)
            plt.scatter(locations, values, color='red')
        if title:
            plt.title(title)
        if save_name:
            plt.savefig(save_name)
        plt.show()
        return None

    """ ================================================================================
    Live plot the data in the data buffer onto a figure and update continuously
    :param: None
    :return: None
    ================================================================================ """
    def plot_live(self, window_size, plot=True):
        ## TO DO ##
        # 1. Clear the plot axes (https://matplotlib.org/3.1.1/api/pyplot_summary.html)
        # 2. Plot the the data buffer (Note: the time buffer is not being plotted!)
        # 3. Show the plot with the argument: block=False
        # 4. Pause the plot for 0.001 seconds
        # 5. Return nothing

        if self.__counter >= window_size:

            buflen = len(self.__data_buffer)
            data = self.__data_buffer[buflen-window_size : buflen]
            time = self.__time_buffer[buflen-window_size : buflen]

            # precompute filter stuff
            order = 2
            wn_lopass = 10.0 / (0.5 * self.__fs)
            wn_hipass = 0.5 / (0.5 * self.__fs)
            b_lopass, a_lopass = sig.butter(order, wn_lopass, 'lowpass')
            b_hipass, a_hipass = sig.butter(order, wn_hipass, 'highpass')
            zi_hipass = sig.lfilter_zi(b_hipass, a_hipass)
            zi_lopass = sig.lfilter_zi(b_lopass, a_lopass)

            data = sig.detrend(data) 
            data = sig.lfilter(b_lopass, a_lopass, data)
            data = sig.lfilter(b_hipass, a_hipass, data)
            data = np.gradient(data)

            self.__data_buffer[buflen-window_size : buflen] = data
            self.__counter = 0

            gmm = self.__model
            data = np.array(data).reshape(-1, 1)
            labels = gmm.predict(data)
            data_norm = (data - data.min())/(data.max() - data.min())

            heart_rate = self.hr_heuristics(time, data, labels)
            # self.__heart_rate = 0.5*(self.__heart_rate + heart_rate)
            self.__heart_rate_list.append(heart_rate)

            # hr_avg = stat.mean(self.__heart_rate_list)
            # self.__heart_rate = hr_avg
            self.__heart_rate = stat.mean(self.__heart_rate_list[-4:])

            if plot==True:
                plt.cla()
                plt.plot(data_norm)
                plt.plot(labels)
                plt.title("Avg Heart Rate: "+("%.2f" % hr_avg)+" BPM")
                plt.xlabel("3 Second Window")
                plt.ylabel("Values")
                plt.show(block=False)
                plt.pause(0.001)

            # return hr_avg

        return self.__heart_rate


    """ ================================================================================
    This function runs the contents of the __data_buffer through a low-pass filter. It
    first generates filter coefficients and  runs the data through the low-pass filter.
    Note: In the future, we will only generate the coefficients once and reuse them.
    :param cutoff: (int) the cutoff frequency of the filter
    :return: None
    ================================================================================ """
    def __lowpass_filter(self, cutoff, order=3): # __ makes this a private method

        ### TO DO ###
        # 1. Use the butter() command from Scipy to produce the filter coefficients
        #    for a 3rd order (N=3) filter and set analog to False. Remember that
        #    'cutoff' must be normalized between 0-1!
        # 2. Filter the data using the lfilter() command
        # 3. Assign the filtered data to the data buffer
        # 4. Return nothing
        b, a = sig.butter(order, cutoff, btype='lowpass', analog=False, output='ba')
        signal_out = sig.lfilter(b, a, self.__data_buffer)
        self.__data_buffer = signal_out
        return


    """ ================================================================================
    This function runs the contents of the __data_buffer through a high-pass filter. It
    first generates filter coefficients and runs the data through the high-pass filter.
    Note: In the future, we will only generate the coefficients once and reuse them.
    :param cutoff: (int) the cutoff frequency of the filter
    :return: None
    ================================================================================ """
    def __highpass_filter(self, cutoff, order=3): # __ makes this a private method

        ### TO DO ###
        # 1. Use the butter() command from Scipy to produce the filter coefficients
        #    for a 3rd order (N=3) filter and set analog to False. Remember that
        #    'cutoff' must be normalized between 0-1!
        # 2. Filter the data using the lfilter() command
        # 3. Assign the filtered data to the data buffer
        # 4. Return nothing

        b, a = sig.butter(order, cutoff, btype='highpass', analog=False, output='ba')
        signal_out = sig.lfilter(b, a, self.__data_buffer)
        self.__data_buffer = signal_out
        return


    """ ================================================================================
    Runs the contents of the __data_buffer through a moving average filter
    :param N: order of the smoothing filter (the filter length = N+1)
    :return: None
    ================================================================================ """
    def __smoothing_filter(self, N):

        ### TO DO ###
        # 1. Create a boxcar window of length M (M = N+1) and normalize it by M
        # 2. Filter the data using the lfilter() command where b is the window and a=1
        # 3. Assign the filtered data to the data buffer
        # 4. Return nothing
        M = N+1
        box = sig.boxcar(M)/M
        signal_out = sig.lfilter(box, 1, self.__data_buffer)
        self.__data_buffer = signal_out
        return


    """ ================================================================================
    Runs the contents of the __data_buffer through a de-meaning filter. 
    :param: None
    :return: None
    ================================================================================ """
    def __demean_filter(self):
        # Compute the mean using a sliding window 
        filtered = sig.detrend(self.__data_buffer)
        self.__data_buffer = filtered
        return


    """ ================================================================================
    The main process block of the ppg. When completed, this will run through the
    filtering operations and heuristic methods to compute and return the step count.
    For now, we will use it as our "playground" to filter and visualize the data.
    :param None:
    :return: Current step count
    ================================================================================ """
    def __filter_ppg(self, name='', plot=False):

        ## TO DO ## 
        # 1. Run the raw data through the demean filter
        # 2. After, run the smoothing_filter with a window of 5
        # 3. Take the gradient of the data using np.gradient 
        # 4. Use a lowpass fiter with a cutoff frequency of around 5Hz, REMEMBER to calculate the normalized cutoff frequency and use that in your function call
        # 5. Return
        if plot:
            self.plot(title=name+' Raw', save_name=name+'Raw')

        self.__demean_filter()
        if plot:
            self.plot(title=name+' Demean', save_name=name+'dm')

        # self.__lowpass_filter(5/(0.5*self.__fs), order=3)
        # if plot:
        #     self.plot(title=name+' Low Pass', save_name=name+'lp')

        # self.__highpass_filter(0.5/(0.5*self.__fs), order=3)
        # if plot:
        #     self.plot(title=name+' High Pass', save_name=name+'hp')

        self.__data_buffer = np.gradient(self.__data_buffer)
        if plot:
            self.plot(title=name+' Gradient', save_name=name+'grad')

        min_max_scaler = preprocessing.MinMaxScaler()
        self.__data_buffer = min_max_scaler.fit_transform(self.__data_buffer[:,np.newaxis]).ravel()

        self.__time_buffer = [ x-self.__time_buffer[0] for x in self.__time_buffer ]

        return

    def __find_heartbeats(self, name=''):
        
        # find peaks
        peak_locations = sig.find_peaks(self.__data_buffer)[0]
        for peak_location in peak_locations:
            peak_time = self.__time_buffer[peak_location]
            peak_value = self.__data_buffer[peak_location]
            if peak_value >= self.__lower_threshold and peak_value <= self.__higher_threshold:
                self.__heartbeats.append( (peak_time, peak_value) )


    def train(self, train_data):
        # 1. Create a GMM object and specify the number of components (classes) in the object
        # 2. Fit the model to our training data. NOTE: You may need to reshape with np.reshape(-1,1) 
        # 3. Return None
        data = np.array(train_data).reshape(-1, 1)
        gmm = GaussianMixture(n_components = 2)
        fit = gmm.fit(data)

        sort_indices = gmm.means_.argsort(axis = 0)
        order = sort_indices[:, 0]
        gmm.means_ = gmm.means_[order,:]
        gmm.covariances_ = gmm.covariances_[order, :]
        w = np.split(gmm.weights_,2)
        w = np.asarray(w)
        w = np.ravel(w[order,:])
        gmm.weights_ = w

        self.__model = gmm

        return
        
    def plt_hist(self, data, bin_count, title=None, save=None):
        # 1. Retrieve Gaussian parameters (these are all attributes of the self.__model)
            # 1.1 Retrieve means for class 0 and class 1 
            # 1.1 Retrieve the covariances for class 0 and 1 and take the square root of each
            # 1.2 Retrieve the weights for class 0 and 1 
        # 2. Create a vector 'x' that will be a vector with a 1000 elements from the min(ir) value to the max(ir) value. HINT: Use np.linspace(). You may need to use np.reshape(x,[1000,1])
        # 3. Compute normal curves for class 0 and 1 HINT: This is done by multiplying the weight by the normalized pdf. Use a function called norm.pdf
        # 4. Create a new figure
        # 5. Plot the histogram      
        # 6. Plot the sum of the normal curvescurves NOTE: Distinguish the two curves by making them different colors.
        # 7. Label the plots HINT: X axis is the PPG reading and y axis the count number.
        # 8. Title the plot "IR Signal Histogram"
        # 9. Call plt.tight_layout()
        # 10. Show the plot and set block to false
        # 11. Create a new figure
        # 12. Plot the histogram
        # 13. Plot each individual curves
        # 14. Label and title the plots
        # 15. Return None
        gmm = self.__model

        mu1 = gmm.means_[0, 0]
        mu2 = gmm.means_[1, 0]
        var1, var2 = gmm.covariances_
        wgt1, wgt2 = gmm.weights_

        print(min(data), max(data), 1000)
        x = np.linspace(min(data), max(data), 1000)
        
        norm1 = wgt1*norm.pdf(x, mu1, np.sqrt(var1).reshape(1))
        norm2 = wgt2*norm.pdf(x, mu2, np.sqrt(var2).reshape(1))

        plt.cla()
        plt.hist(data, bins=bin_count, density=True, alpha=.3, label='Histogram')    
        plt.plot(x, norm1+norm2, label='sum')
        plt.legend(loc='best', fontsize='x-small')
        plt.xlabel('PPG Reading (Normalized)')
        plt.ylabel('Count (Normalized)')
        plt.title("IR Signal Histogram \n"+title)
        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.show(block=False)

        plt.cla()
        plt.hist(data, bins=bin_count, density=True, alpha=.3, label='Histogram')    
        plt.plot(x, norm1, label='norm1')
        plt.plot(x, norm2, label='norm2')
        plt.legend(loc='best', fontsize='x-small')
        plt.xlabel('PPG Reading (Normalized)')
        plt.ylabel('Count (Normalized)')
        plt.title("IR Signal Histogram \n"+title)
        plt.tight_layout()
        if save:
            plt.savefig(save)
        plt.show(block=False)

        return

    def plt_labels(self, data, time, window=None, title=None, save=None):
        # 1. Calculate the labels by running the IR data through the model
        # 2. Create a new figure
        # 3. Plot the time and IR data 
        # 4. Plot the time and label data
        # 5. Label the x and y axes
        # 6. Title the plot "GMM Labels"
        # 7. Show the plot and set block to false
        # 8. Return None
        gmm = self.__model

        plt.cla()
        data = np.array(data).reshape(-1, 1)
        labels = gmm.predict(data)
        if window:
            plt.plot(time[window], data[window], label='Voltage')
            plt.plot(time[window], labels[window], color='red', label='Prediction')
        else:
            plt.plot(time, data, label='Voltage')
            plt.plot(time, labels, color='red', label='Prediction')
        plt.xlabel('Time')
        plt.ylabel('Reading')
        plt.title("GMM Prediction Labels \n "+title)
        if save:
            plt.savefig(save)
        plt.show(block=False)

        return

    def hr_heuristics(self, time, data, labels):
        # At a bare minimum, you can do the following steps. BUT you should definitely consider the edge cases we discussed in class!!!
        # 1. Find the indices of the maximum of the peaks from the labels 
        # 2. Next compute the time difference between all of the peaks 
        # 3. Prune the outlier peaks
            # 3.1 For example, you coul loop through the time differences and check if the length is above a threshold
            # 3.2 If it is below the threshold, then discard it as a false reading
            ## Remember: You could try a low-pass filter instead of doing time thresholding. It's your choice which one you prefer. Get creative!
        # 4. Get the total count of valid heartbeat times from the labels
        # 5. Calculate the estimated heart rate given the valid beats
        # 6. Store the HR in your HR buffer
        # 7. Return the current HR

        # print(len(time), len(data))

        # use length of peak heuristic using scipy instead of loop -> better
        width_threshold=12
        peak_locations = sig.find_peaks(labels, distance=width_threshold)[0]

        # print(peak_locations)

        # for peak_location in peak_locations:
        #     peak_time = time[peak_location]
        #     peak_value = data[peak_location]
        #     peak_label = labels[peak_location]
        #     print(peak_time, peak_value, peak_label)


        # window = slice(0, 300)
        # plt.cla()
        # plt.plot(time[window], data[window], label='Voltage')
        # plt.plot(time[window], labels[window], color='red', label='Prediction')
        # plt.legend(loc='best', fontsize='x-small')
        # plt.xlabel('Time')
        # plt.ylabel('Reading')
        # plt.title("HR Heuristics")
        # plt.show()
        
        heart_rate = len(peak_locations) / (time[-1]-time[0])
        heart_rate = heart_rate * 10e2 # get beats per second
        heart_rate = heart_rate * 60 # get beats per minute
        self.__heart_rate = heart_rate
        return heart_rate

    def filter_ppg(self): 
        return self.__filter_ppg()
    def get_time(self): 
        return self.__time_buffer
    def get_data(self):
        return self.__data_buffer
        
    def process(self, time, data):
        # 1. Using the GMM model from the previous objective, call the GMM's predict() method to classify the new data
        # 2. Next call hr_heuristics() to calculate the heart rate 
        # 3. Return the HR

        gmm = self.__model
        data = np.array(data).reshape(-1, 1)
        labels = gmm.predict(data)

        heart_rate = self.hr_heuristics(time, data, labels)
        return heart_rate

        
        
