"""
Author: Ramsin Khoshabeh
Contact: ramsin@ucsd.edu
Date: 07 April 2019

Description: An entry point for testing the PPG class
"""

import traceback
import time

from mywearable.ble import BLE
from mywearable.ppg import PPG

""" -------------------- Settings -------------------- """
run_config = False                # whether to config PC HM-10 or not
baudrate = 9600                   # PySerial baud rate of the PC HM-10
serial_port = 'COM3'               # Serial port of the PC HM-10
peripheral_mac = '78DB2F13E9D2'   # Mac Address of the Arduino HM-10

signal_len = 30  # length of signal in seconds (start with 10)
sample_rate = 25                  # samples / second
buff_len = signal_len*sample_rate # length of the data buffers
plot_refresh = 50                 # draw the plot every X samples (adjust as needed)

""" -------------------- Test #1 -------------------- """
ppg = PPG(buff_len, sample_rate)

hm10 = BLE(serial_port, baudrate, run_config)
hm10.connect(peripheral_mac)

try:
    counter = 0
    for i in range(10):
        msg = hm10.read_line(';') # avoid junk data
    while(True):
        msg = hm10.read_line(';')
        print(msg)
        hm10.write('test')
        if len(msg) > 0:
            retval = ppg.append(msg)
            if retval == False: 
              ppg.save_file("PPG.csv")
              break

            if counter % plot_refresh == 0:
                ppg.plot_live()
            counter += 1
except KeyboardInterrupt:
    print("\nExiting due to user input (<ctrl>+c).")
    hm10.close()
except Exception as e:
    print("\nExiting due to an error.")
    traceback.print_exc()
    hm10.close()

""" -------------------- Test #2 -------------------- """
# files = ['PPGRaw1.csv', 'PPGRaw2.csv', 'PPGRaw3.csv', 'PPGRaw4.csv', 'PPGRaw5.csv']
# names = ['PPG1', 'PPG2', 'PPG3', 'PPG4', 'PPG5']
# for index, file in enumerate(files):
#     ppg = PPG(buff_len, sample_rate)
#     ppg.load_file(file)
#     ppg.process(name=names[index])
#     # ppg.plot()
