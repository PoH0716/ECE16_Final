"""
Author: Curtis, Po
Contact: don't
Date: today

Description: An entry point for the final project
"""

import traceback
import time
import glob
import operator
import pyautogui

from gmm_test import get_subjects
from gmm_test import load_files

from mywearable.ble import BLE
from mywearable.ppg import PPG
from mywearable.pedometer import Pedometer

from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from scipy import integrate


""" -------------------- Settings -------------------- """
run_config = False                # whether to config PC HM-10 or not
baudrate = 9600                   # PySerial baud rate of the PC HM-10
serial_port = 'COM10'               # Serial port of the PC HM-10
peripheral_mac = '78DB2F13E9D2'   # Mac Address of the Arduino HM-10
# peripheral_mac = '78DB2F168240'   # Mac Address of the Arduino HM-10

signal_len = 30  # length of signal in seconds (start with 10)
sample_rate = 25                  # samples / second
buff_len = signal_len*sample_rate # length of the data buffers
# buff_len = 10
plot_refresh = 50                 # draw the plot every X samples (adjust as needed)

class Circular():
  def __init__(self, size):
    self.values = list()
    self.size = size

  def append(self, value):
    self.values.append(value)
    while len(self.values) > self.size:
      self.values.pop(0)
    assert len(self.values) <= self.size
  
  def length(self):
    return len(self.values)

def filter_pedo(data, N, b_lopass, a_lopass):
  print(list(data))
  M = N+1
  box = sig.boxcar(M)/M
  #__demean_filter
  data = sig.detrend(data) 
  #__smoothing_filter
  data = sig.lfilter(box, 1, data)
  #__gradient_filter
  data = np.gradient(data)
  #__lowpass_filter
  data = sig.lfilter(b_lopass, a_lopass, data)
  return data

def filter_ppg(data, N, b_lopass, a_lopass, b_hipass, a_hipass, z_lo=None, z_hi=None):
  print(list(data))
  zo_lo = None
  zo_lo = None
  # M = N+1
  # box = sig.boxcar(M)/M
  #__demean_filter
  data = sig.detrend(data) 
  #__lowpass_filter
  try:
    data, zo_lo = sig.lfilter(b_lopass, a_lopass, data, zi=z_lo)
  except:
    data = sig.lfilter(b_lopass, a_lopass, data)
  #__highpass_filter
  # data, zo_hi = sig.lfilter(b_hipass, a_hipass, data, zi=z_hi)
  #__gradient_filter
  data = np.gradient(data)

  # MinMaxScaler
  # min_max_scaler = preprocessing.MinMaxScaler()
  # data = min_max_scaler.fit_transform(data[:,np.newaxis]).ravel()

  #__smoothing_filter
  # data = sig.lfilter(box, 1, data)
  return data, zo_lo

""" ================================================================================
Saves the contents of the buffer into the specified file one line at a time.
:param filename: (str) the name of the file that will store the buffer data
:return: None
================================================================================ """
def save_file(filename, buffer):

    ### TO DO ###
    # 1. Open the file with mode set to write (this will overwrite any data)
    # 2. In a for-loop, iterate over the buffers
        # 2.1. Assemble a "row" string from the buffers formatted as "<t>,<d>\n"
        # 2.2. Write the row to the file
    # 3. Close the file or use the "with" keyword
    # 4. Return nothing

    # NOTE: This method is the same as in the Pedometer class
    f = open(filename, "w")
    for group in buffer:
        group = group.strip(';')
        time, pedo, ppg = list(map(int, group.split(','))) 
        row_str = "%d,%d,%d\n" % (time, pedo, ppg)
        f.write(row_str)
    f.close()
    return None

""" -------------------- Test #2 -------------------- """
pedo = Pedometer(buff_len, False)
hm10 = BLE(serial_port, baudrate, run_config)
# hm10.connect(peripheral_mac)

buf_siz = 3

accel_start = None
vel_start = None
accel_buf = Circular(buf_siz)
vel_buf = Circular(buf_siz)
pos = (0, 0, 0)

last_time = 0.0
current_time = 0.0

last_accel = None

accel_begin = None
threshold = 8000
debounce = 0.15

try:
    counter = 0
    for i in range(10):
        msg = hm10.read_line(';') # avoid junk data
    while(True):
      try:
        msg = hm10.read_line(';')

        data = tuple(map(lambda x: int(x.strip()), msg.split(',')))
        accel = data[1]
        if not accel_begin: accel_begin = accel
        if not last_accel: last_accel = accel

        current_time = time.time()
        if accel-accel_begin < -threshold:
          if current_time - last_time > debounce:
            print(accel-accel_begin)
            print('click')
            pyautogui.click(button='left', clicks=1)
          last_time = current_time

      # except ValueError:
      #   continue
      # except pyautogui.FailSafeException:
      #   continue
      except:
        continue
except KeyboardInterrupt:
    print("\nExiting due to user input (<ctrl>+c).")
    hm10.close()
except Exception as e:
    print("\nExiting due to an error.")
    traceback.print_exc()
    hm10.close()