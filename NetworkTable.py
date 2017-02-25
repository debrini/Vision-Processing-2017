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

# declare it
#NetworkTable table

# Init NetworkTable
NetworkTable.setClientMode()
NetworkTable.setTeam(245)
NetworkTable.setIPAddress("10.2.45.23") # ip of roborio
NetworkTable.initialize()
table = NetworkTable.getTable("visionTable") # what table data is put in

# in the processing
visionTable.putNumber('area', area)
visionTable.putNumber('dist_between', dist_between)
