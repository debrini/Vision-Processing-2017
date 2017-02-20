#!/usr/bin/env python2
#
# This is a NetworkTables client (eg, the DriverStation/coprocessor side).
# You need to tell it the IP address of the NetworkTables server (the
# robot or simulator).
#
# When running, this will continue incrementing the value 'dsTime', and the
# value should be visible to other networktables clients and the robot.

import sys
import time
from networktables import NetworkTable

# To see messages from networktables, you must setup logging
import logging
logging.basicConfig(level=logging.DEBUG)
'''
if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1] # when calling this program, write name (space) IP address?
'''
ip = sysargv[10.51.4.66]
NetworkTable.setIPAddress("10.2.45.2")#Your IP goes here


NetworkTable.initialize(server=ip)

vp = NetworkTables.getTable("VisionProcessing")


while True:
	try:
		print('robotTime:', sd.getNumber('robotTime'))
	except KeyError:
		print('robotTime: N/A')

	vp.putString('go_forward_and_left', go_forward_and_left)
	vp.putString('go_forward_and_right', go_forward_and_right)
	vp.putString('execute_gear_drop_off', execute_gear_drop_off)
	vp.putString('reverse', reverse)
	vp.putString('stop', stop)

	time.sleep(1)
