# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from os import read
import vrep 
import sys
import time 
import numpy as np
from tank import *


# %%
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * distance and velocity on subjective ranges [0.7, 5.0], [0, 10]
#   * v has a range of [0, 10] in units of percentage points
x_dist = np.arange(0.7, 5.1, 0.1)
x_velo = np.arange(0, 10.1, 0.1)
x_v  = np.arange(0, 10.1, 0.1)

# Generate fuzzy membership functions
dist_lo = fuzz.trimf(x_dist, [0.7, 0.7, 3.0])
dist_md = fuzz.trimf(x_dist, [0.7, 3.0, 5.1])
dist_hi = fuzz.trimf(x_dist, [3.0, 5.1, 5.1])
velo_lo = fuzz.trimf(x_velo, [0, 0, 5])
velo_md = fuzz.trimf(x_velo, [0, 5, 10])
velo_hi = fuzz.trimf(x_velo, [5, 10, 10])
v_lo = fuzz.trimf(x_v, [0, 0, 5])
v_md = fuzz.trimf(x_v, [0, 5, 10])
v_hi = fuzz.trimf(x_v, [5, 10, 10])

def fuzzy(velo, dist):
    dist_level_lo = fuzz.interp_membership(x_dist, dist_lo, dist)
    dist_level_md = fuzz.interp_membership(x_dist, dist_md, dist)
    dist_level_hi = fuzz.interp_membership(x_dist, dist_hi, dist)

    velo_level_lo = fuzz.interp_membership(x_velo, velo_lo, velo)
    velo_level_md = fuzz.interp_membership(x_velo, velo_md, velo)
    velo_level_hi = fuzz.interp_membership(x_velo, velo_hi, velo)


    # Now we take our rules and apply them. Rule 1 concerns bad food OR service.
    # The OR operator means we take the maximum of these two.
    lo_rule1 = np.fmin(dist_level_md, velo_level_hi)
    lo_rule2 = np.fmin(dist_level_lo, v_lo)
    lo_rule = np.fmax(lo_rule1, lo_rule2)

    md_rule = np.fmin(dist_level_md, v_md)

    hi_rule1 = np.fmin(dist_level_md, velo_level_lo)
    hi_rule2 = np.fmin(dist_level_hi, v_hi)
    hi_rule = np.fmax(hi_rule1, hi_rule2)

    v_activation_lo = np.fmin(lo_rule, v_lo)  
    v_activation_md = np.fmin(md_rule, v_md)
    v_activation_hi = np.fmin(hi_rule, v_hi)
    aggregated = np.fmax(v_activation_lo,
                        np.fmax(v_activation_md, v_activation_hi))
    return fuzz.defuzz(x_v, aggregated, 'lom') #centroid


# %%
vrep.simxFinish(-1) # closes all opened connections, in case any prevoius wasnt finished
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # start a connection

if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

#create instance of Tank
tank=Tank(clientID)


# %%



# %%
proximity_sensors=["NW"]#,"NW"]
proximity_sensors_handles=[0]*len(proximity_sensors)

# get handle to proximity sensors
for i in range(len(proximity_sensors)):
    err_code,proximity_sensors_handles[i] = vrep.simxGetObjectHandle(clientID,"Proximity_sensor_"+proximity_sensors[i], vrep.simx_opmode_blocking)
    
#read and print values from proximity sensors
#first reading should be done with simx_opmode_streaming, further with simx_opmode_buffer parameter
for sensor_name, sensor_handle in zip(proximity_sensors,proximity_sensors_handles):
        err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)


# %%
tank.forward(10)
# tank.rightVelocity - velocity
#continue reading and printing values from proximity sensors
t = time.time()
run = True
while run: # read values for 5 seconds
    for sensor_name, sensor_handle in zip(proximity_sensors,proximity_sensors_handles):
        err_code,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_buffer )
        if(err_code == 0):
            reading = np.linalg.norm(detectedPoint)
            print(sensor_name, reading, end=" ")
            if reading > 5.0:
                continue
            velo = fuzzy(tank.rightvelocity, reading)
            print("Velo", velo)
            if velo <= 0.00001:
                print("Ending")
                run = False
            tank.forward(velo)
    # print()


# %%



