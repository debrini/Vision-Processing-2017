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
NetworkTable table

# Init NetworkTable
NetworkTable.setClientMode()
NetworkTable.setTeam(245)
NetworkTable.setIPAddress("10.2.45.23") # ip of roborio
NetworkTable.initialize()
table = NetworkTable.getTable("PiTable") # what table data is put in

# in the processing
table.putString('go_forward_and_left', go_forward_and_left())
table.putString('go_forward_and_right', go_forward_and_right())
table.putString('execute_gear_drop_off', execute_gear_drop_off())
table.putString('reverse', reverse())
table.putString('stop', stop())
