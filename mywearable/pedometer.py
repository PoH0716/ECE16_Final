"""
Authors: Avak Archanian / Ramsin Khoshabeh / Edward Wang
Contact: aarchani@ucsd.edu / ramsin@ucsd.edu / ejaywang@eng.ucsd.edu
Date: 25 October 2019

Description: A class to handle the pedometer for the wearable
"""

# Imports
import serial
from time import sleep
from time import time
from scipy import signal as sig
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig

class Pedometer:

    # Attributes of the class Pedometer
    _maxlen = 0
    _file_flag = False
    __time_buffer = [] # private variable (encapsulation)
    __data_buffer = [] # private variable (encapsulation)
    __steps = 0
    __freq_sampling = 33
    __counter = 0

    """ ================================================================================
    Constructor that sets up the Pedomter class. It will only run once.
    :param max len: (int) max length of the buffer
    :param file_flag: (bool) set whether we will be working with a file or not
    :return: None
    ================================================================================ """
    def __init__(self, maxlen, file_flag):
        self._maxlen = maxlen       # Set the max length of the buffer
        self._file_flag = file_flag # Set whether we are writing to a file or not
        return

    """ ================================================================================
    Resets the pedometer to default state
    :return: None
    ================================================================================ """
    def reset(self):
        self._maxlen = 0
        self.__time_buffer = []
        self.__data_buffer = []
        __steps = 0
        return

    """ ================================================================================
    Appends new elements to the data and time buffers by parsing 'msg_str' and splitting
    it, assuming comma separation. It also keeps track of buffer occupancy and notifies
    the user when the buffers are full.
    :param msg_str: (str) the string containing data that will be appended to the buffer
    :return: None
    ================================================================================ """
    def append(self, msg_str):

        ### TO DO ###
        # 1. Check if the length of one of buffers (both should be the same) is equal to maxlen
            # 1.1. If they are equal, print a message that the buffer is full
            # 1.2. do nothing and return
        # if len(self.__time_buffer) == self._maxlen:
        #     print("Buffer is full!")
        #     return False
        # 2. Inside of a try-except block
            # 2.1. Split the incoming message by looking for ','
            # 2.2. Append the time to the time buffer (make sure to cast as an int)
            # 2.3. Append the data to the data buffer (make sure to cast as an int)
        # 3. If there is a ValueError exception
            # print that there was invalid data
        
        retval = True
        while len(self.__time_buffer) >= self._maxlen:
            self.__time_buffer.pop(0)
            self.__data_buffer.pop(0)
            retval = False        
        
        try:
            result = [int(x.strip()) for x in msg_str.split(',')]
            self.__time_buffer.append(result[0])
            self.__data_buffer.append(result[1])
            self.__counter += 1
        except ValueError:
            print("There wuz invalid data!")
        except IndexError:
            pass
        # 4. Return True if the buffer is not full yet, False otherwise
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
        f = open(filename, "w")
        combined_buffer = list(zip(self.__time_buffer, self.__data_buffer))
        for point in combined_buffer:
            row_str = "%d,%d\n" % (point[0], point[1])
            f.write(row_str)
        f.close()
        return None


    """ ================================================================================
    Loads the contents of the file 'filename' into the time and data buffers 
    :param filename: (str) the name (full path) of the file that we read from
    :return: None
    ================================================================================ """
    def load_file(self, filename):
        self.__time_buffer = []
        self.__data_buffer = []

        ### TO DO ###
        # 1. Open the file with mode set to read
        # 2. Loop forever
            # 2.1. Read a line from the file 
            # 2.2. Check to see if the line is None
                #2.2.1. If it is, break out of the loop (the end of file was reached)
            # 2.3. Strip any newline characters from the line and split it with ','
            # 2.4. Append time value to the time buffer
            # 2.5. Append data value to the data buffer
            # 2.6. Keep track of the number of times the while loop runs in a variable
        # 3. Set the object's _maxlen (self._maxlen) to the loop count variable
        # 4. Return nothing
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
    def plot(self, title=None, save_name=None):
        ## TO DO ##
        # 1. Open a new figure (if needed)
        # 2. Plot the time buffer versus the data buffer (stylize however you like)
        # 3. Show the plot
        # 4. Return nothing
        plt.plot(self.__time_buffer, self.__data_buffer)
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
            M = 6
            box = sig.boxcar(M)/M
            order = 2
            wn_lopass = 4.0 / (0.5 * self.__freq_sampling)
            wn_hipass = 0.25 / (0.5 * self.__freq_sampling)
            b_lopass, a_lopass = sig.butter(order, wn_lopass, 'lowpass')
            b_hipass, a_hipass = sig.butter(order, wn_hipass, 'highpass')
            zi_hipass = sig.lfilter_zi(b_hipass, a_hipass)
            zi_lopass = sig.lfilter_zi(b_lopass, a_lopass)

            # __filter_pedometer
            data = sig.detrend(data) 
            # data = sig.lfilter(box, 1, data)
            data = sig.lfilter(b_lopass, a_lopass, data)
            # data = sig.lfilter(b_hipass, a_hipass, data)
            data = np.gradient(data)

            peaks = sig.find_peaks(data)

            lower_threshold = 1800
            higher_threshold = 5000
            peak_index_list = peaks[0]

            steps = 0
            inds = list()
            peaks = list()
            print(peak_index_list)
            for index in peak_index_list:
                peak = data[index]
                if peak > lower_threshold and peak < higher_threshold:
                    steps += 1   
                    inds.append( index )
                    peaks.append( peak )

            if plot==True:
                plt.cla()
                plt.plot(data)
                plt.scatter(inds, peaks)
                plt.title("Pedometer")
                plt.xlabel("3 Second Window")
                plt.ylabel("Values")
                plt.ylim(0, 10000)
                plt.show(block=False)
                plt.pause(0.001)

            # return hr_avg

            self.__steps += steps
        return self.__steps

    """ ================================================================================
    This function runs the contents of the __data_buffer through a low-pass filter. It
    first generates filter coefficients and  runs the data through the low-pass filter.
    Note: In the future, we will only generate the coefficients once and reuse them.
    :param cutoff: (int) the cutoff frequency of the filter
    :return: None
    ================================================================================ """
    def __lowpass_filter(self, cutoff): # __ makes this a private method

        ### TO DO ###
        # 1. Use the butter() command from Scipy to produce the filter coefficients
        #    for a 3rd order (N=3) filter and set analog to False. Remember that
        #    'cutoff' must be normalized between 0-1!
        # 2. Filter the data using the lfilter() command
        # 3. Assign the filtered data to the data buffer
        # 4. Return nothing
        b, a = sig.butter(3, cutoff, btype='lowpass', analog=False, output='ba')
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
    def __highpass_filter(self, cutoff): # __ makes this a private method

        ### TO DO ###
        # 1. Use the butter() command from Scipy to produce the filter coefficients
        #    for a 3rd order (N=3) filter and set analog to False. Remember that
        #    'cutoff' must be normalized between 0-1!
        # 2. Filter the data using the lfilter() command
        # 3. Assign the filtered data to the data buffer
        # 4. Return nothing

        b, a = sig.butter(3, cutoff, btype='highpass', analog=False, output='ba')
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
    The main process block of the pedometer. When completed, this will run through the
    filtering operations and heuristic methods to compute and return the step count.
    For now, we will use it as our "playground" to filter and visualize the data.
    :param None:
    :return: Current step count
    ================================================================================ """
    def __filter_pedometer(self):

        ## TO DO ## 
        # 1. Run the raw data through the demean filter
        # 2. After, run the smoothing_filter with a window of 5
        # 3. Take the gradient of the data using np.gradient 
        # 4. Use a lowpass fiter with a cutoff frequency of around 5Hz, REMEMBER to calculate the normalized cutoff frequency and use that in your function call
        # 5. Return
        self.__demean_filter()
        self.__smoothing_filter(5)
        self.__data_buffer = np.gradient(self.__data_buffer)
        self.__lowpass_filter(3/(0.5*self.__freq_sampling))
        return

    def __find_peaks(self):

        ## TO DO ## 
        # 1. Initialize self.__peaks to an empty list 
        # 2. Filter all of the data in the self.__data_buffer (HINT: you wrote a method that does this!)
        # 3. Find the indices of all the peaks above the 0 threshold and store them in self.__peaks. (HINT: try scipy.signal.find_peaks)
        # 4. Return
        self.__peaks = list()
        self.__filter_pedometer()
        self.__peaks = sig.find_peaks(self.__data_buffer)
        return

    """ ================================================================================
    Saves the contents of the buffer into the file line by line 
    :param filename: (str) the name of the file that will store the buffer data
    :return: None
    ================================================================================ """
    def __count_steps(self):

        ## TO DO ## 
        # 1. initialize variable inds to be an empty list
        # 2. Parse through all the indices in self.__peaks
            # 2.1. check to see if the value of self.__data_buffer at the index of the peak is greater than the lower threshold and smaller than the upper threshold 
                # 2.1.1. increment the step counter self.__steps
                # 2.1.2. appeand that peak into inds 
        # 3. plot the contents of self.__data_buffer[inds] (DON'T USE self.plot() THIS IS FOR YOUR REFERRENCE ONLY! Write it in __count_steps and then comment it out!!!!)

        lower_threshold = 400
        higher_threshold = 4000

        # inds = list()
        # peaks = list()
        peak_index_list = self.__peaks[0]
        for index in peak_index_list:
            peak = self.__data_buffer[index]
            if peak > lower_threshold and peak < higher_threshold:
                self.__steps += 1
                # inds.append( index )
                # peaks.append( peak )
        
        # print(inds)
        # plt.plot(self.__data_buffer)
        # plt.scatter(inds, peaks)
        # plt.show()
        return
    
    
    def objective2graphgen(self):
        freq_data = self.__freq_sampling
        freq_cutoff = [0.01, 0.5, 1, 5, 10, 15]
        freq_savename = ['001', '05', '1', '5', '10', '15']
        freq_cutoff_norm = [ x/(0.5*freq_data) for x in freq_cutoff ]
        for index, freq in enumerate(freq_cutoff_norm):
            self.reset()
            self.load_file('walking_33hz.txt')
            self.__lowpass_filter(freq)
            self.plot(
                title='Low Pass '+str(freq_cutoff[index])+'Hz', 
                save_name='IMU_filtered_LPF'+freq_savename[index])
        for index, freq in enumerate(freq_cutoff_norm):
            self.reset()
            self.load_file('walking_33hz.txt')
            self.__highpass_filter(freq)
            self.plot(
                title='High Pass '+str(freq_cutoff[index])+'Hz',
                save_name='IMU_filtered_HPF'+freq_savename[index])
        self.reset()
        self.load_file('walking_33hz.txt')
        self.__demean_filter()
        self.plot(
            title='Demean Filter',
            save_name='IMU_filtered_DM')
        smoothing_m = [1, 5, 12, 30, 60]
        for M in smoothing_m:
            self.reset()
            self.load_file('walking_33hz.txt')
            self.__smoothing_filter(M)
            self.plot(
                title='Smoothing Filter M='+str(M),
                save_name='IMU_filtered_SM'+str(M))
    
    def extracreditgraphgen(self):
        freq_data = self.__freq_sampling
        self.reset()
        self.load_file('walking_33hz.txt')
        self.__smoothing_filter(5)
        self.plot(
            title='Detrend Appromimation M=5',
            save_name='ExtraCreditDetrend')

        self.reset()
        self.load_file('walking_33hz.txt')
        self.__lowpass_filter(0.2)
        self.plot(
            title='Detrend Appromimation 0.2Hz',
            save_name='ExtraCreditHigh-Pass')
    


    def process(self):
        
        ### TO DO ###
        # 1. Load the file walking_50hz.txt
        # 2. Plot it

        # 3. Low-pass filter the signal
        # 4. Plot it

        # 5. Reset the buffers
        # 6. Reload the file
        # 7. High-pass filter the signal
        # 8. Plot it

        # 9. Reset the buffers
        # 10. Reload the file
        # 11. Smoothing filter the signal
        # 12. Plot it

        # 13. Reset the buffers
        # 14. Reload the file
        # 15. De-mean the signal
        # 16. Plot it

        # 17. For low-pass & high-pass filters only, loop over the following cutoffs:
        #     0.01 Hz, 0.1 Hz, 0.5 Hz, 1 Hz, 10 Hz, 15 Hz
        #   17.1. Reload the file
        #   17.2. Filter the signal with the cutoff
        #   17.3. Plot it

        # 18. Save all your plots (screenshots are fine). You should have 15 in total.

        # objective 2
        # self.objective2graphgen()
        # self.extracreditgraphgen()
        
        ## TO DO ## (BEGIN AT THE START OF OBJECTIVE 3)
        # 1. load the contents of 'walking_50hz.txt' into the data buffer
        # 2. find the peaks of the data
        # 3. Use subplots to plot the raw data and filtered data side-by-side (COMMENT THIS LINE OUT AFTER OBJECTIVE 3!!!!)
        # 3. count the number of steps (DO NOT WRITE THIS LINE UNTIL OBJECTIVE 4!!!!!!!!)
        # 4. Return the number of steps (DO NOT WRITE THIS LINE UNTIL OBJECTIVE 4!!!!!!!!)

        # objective 3 & 4
        # self.reset()
        # self.load_file('walking_33hz.txt')
        # plt.plot(self.__time_buffer, self.__data_buffer)
        self.__find_peaks()
        # plt.plot(self.__time_buffer, self.__data_buffer)
        # plt.show()

        self.__count_steps()
        
        return self.__steps