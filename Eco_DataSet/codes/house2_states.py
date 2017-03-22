# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:03:30 2017

@author: niulongjia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.fftpack

def convertDate(date_string):
    date_list = date_string.split("-")
    day=date_list[0]
    month=date_list[1]
    year=date_list[2]
    date_out=year
    if month=="Jan":
        date_out = date_out+"-01-"
    elif month=="Feb":
        date_out = date_out+"-02-"
    elif month=="Mar":
        date_out = date_out+"-03-"
    elif month=="Apr":
        date_out = date_out+"-04-"
    elif month=="May":
        date_out = date_out+"-05-"
    elif month=="Jun":
        date_out = date_out+"-06-"
    elif month=="Jul":
        date_out = date_out+"-07-"
    elif month=="Aug":
        date_out = date_out+"-08-"
    elif month=="Sep":
        date_out = date_out+"-09-"
    elif month=="Oct":
        date_out = date_out+"-10-"
    elif month=="Nov":
        date_out = date_out+"-11-"
    else:
        date_out = date_out+"-12-"
    date_out = date_out + day
    return date_out


occupancy_fname_suffix="../house#02/02_occupancy_csv/"
plug_fname_suffix="../house#02/02_plugs_csv/"
df_read=pd.read_csv(occupancy_fname_suffix + "02_summer.csv")

index=7
occupancy_day = df_read.iloc[index,1:]
date_string=df_read.iloc[index,0]
date_num=convertDate(date_string)
print date_string + "  " + date_num

####################### Below are showing up appliances states graphs ###########################
# This is for Dishwasher/Kettle
thres_dishwasher_kettle = 100
house2_dishwasher_fname= plug_fname_suffix + "02/" + date_num + ".csv"
house2_kettle_fname= plug_fname_suffix + "07/" + date_num + ".csv"

if os.path.isfile(house2_dishwasher_fname) == True and os.path.isfile(house2_kettle_fname) == True:
    ax1 = plt.subplot(5,1,1)
    df_house2_dishwasher_power = pd.read_csv(house2_dishwasher_fname,names=['power_consumption'])
    df_house2_kettle_power = pd.read_csv(house2_kettle_fname,names=['power_consumption'])
    
    house2_dishwasher_power = df_house2_dishwasher_power.iloc[:,0]
    house2_kettle_power = df_house2_kettle_power.iloc[:,0]
    
    house2_dk_state = house2_dishwasher_power
    for i in range(0,86400):
        if house2_dishwasher_power[i] > thres_dishwasher_kettle or house2_kettle_power[i] > thres_dishwasher_kettle:
            house2_dk_state[i] = 1
        else:
            house2_dk_state[i] = 0
    house2_dk_state.plot(grid=True,legend=True,label='Dishwasher/Kettle State', color='r',linewidth=3)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    

# This is for Entertainment
thres_entertain = 100
house2_entertain_fname= plug_fname_suffix + "05/" + date_num + ".csv"
if os.path.isfile(house2_entertain_fname) == True:
    ax1 = plt.subplot(5,1,2)
    df_house2_entertain_power = pd.read_csv(house2_entertain_fname,names=['power_consumption'])
    house2_entertain_power = df_house2_entertain_power.iloc[:,0]
    house2_entertain_state = house2_entertain_power
    for i in range(0,86400):
        if house2_entertain_power[i] == -1:
            house2_entertain_state[i] = 0
        elif house2_entertain_power[i] > thres_entertain:
            house2_entertain_state[i] = 1
        else:
            house2_entertain_state[i] = 0
    house2_entertain_state.plot(grid=True,legend=True,label='Entertainment State', color='r',linewidth=3)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines

# This is for Lamp 
thres_lamp = 100 # different states ???
house2_lamp_fname= plug_fname_suffix + "08/" + date_num + ".csv"
if os.path.isfile(house2_lamp_fname) == True:
    ax1 = plt.subplot(5,1,3)
    df_house2_lamp_power = pd.read_csv(house2_lamp_fname,names=['power_consumption'])
    house2_lamp_power = df_house2_lamp_power.iloc[:,0]
    house2_lamp_state = house2_lamp_power
    for i in range(0,86400):
        if house2_lamp_power[i] == -1:
            house2_lamp_state[i] = 0
        elif house2_lamp_power[i] > thres_lamp:
            house2_lamp_state[i] = 1
        else:
            house2_lamp_state[i] = 0
    house2_lamp_state.plot(grid=True,legend=True,label='Lamp State', color='r',linewidth=3)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
# This is for laptop 
thres_laptop = 10 # 
house2_laptop_fname= plug_fname_suffix + "09/" + date_num + ".csv"
if os.path.isfile(house2_laptop_fname) == True:
    ax1 = plt.subplot(5,1,4)
    df_house2_laptop_power = pd.read_csv(house2_laptop_fname,names=['power_consumption'])
    house2_laptop_power = df_house2_laptop_power.iloc[:,0]
    house2_laptop_state = house2_laptop_power
    for i in range(0,86400):
        if house2_laptop_power[i] == -1:
            house2_laptop_state[i] = 0
        elif house2_laptop_power[i] > thres_laptop:
            house2_laptop_state[i] = 1
        else:
            house2_laptop_state[i] = 0
    house2_laptop_state.plot(grid=True,legend=True,label='Laptop State', color='r',linewidth=3)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines


ax0 = plt.subplot(5,1,5)
occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
ax0.set_ylim([0,1.005])
####################### Below are building up prediction states ###########################
# e.g 0001 means laptop state is on, other states are off #
time_interval=15 # 15min
thres=60 # if an appliance is active for more than thres(seconds), then prediction state set to 1
slice_interval=time_interval*60
slice_num=24*60/time_interval
appliance_states=np.zeros([slice_num,4]) # 4 prediction states
for i in range(slice_num):
    count_dk = 0
    count_entertainment = 0
    count_lamp = 0
    count_laptop = 0
    for j in range(i*slice_interval, (i+1)*slice_interval):
        if house2_dk_state[j]==1:
            count_dk += 1
        if house2_entertain_state[j]==1:
            count_entertainment += 1
        if house2_lamp_state[j]==1:
            count_lamp += 1
        if house2_laptop_state[j]==1:
            count_laptop += 1
    
    if (count_dk>thres):
        appliance_states[i][0]=1
        
    if (count_entertainment>thres):
        appliance_states[i][1]=1
        
    if (count_lamp>thres):
        appliance_states[i][2]=1
        
    if (count_laptop>thres):
        appliance_states[i][3]=1

####################### Below are showing up power consumption graphs ###########################
house1_sm_fname="../house#02/02_sm_csv/" + date_num + ".csv"
if os.path.isfile(house1_sm_fname)==True:
    df_sm=pd.read_csv(house1_sm_fname,
                      names=['powerallphases',
                             'powerl1',
                             'powerl2',
                             'powerl3',
                             'currentneutral',
                             'currentl1',
                             'currentl2',
                             'currentl3',
                             'voltagel1',
                             'voltagel2',
                             'voltagel3',
                             'phaseanglevoltagel2l1',
                             'phaseanglevoltagel3l1',
                             'phaseanglecurrentvoltagel1',
                             'phaseanglecurrentvoltagel2',
                             'phaseanglecurrentvoltagel3'
                                            ])
    house1_sm_powerallphases = df_sm.iloc[:,0]
    house1_sm_powerl1 = df_sm.iloc[:,1]
    house1_sm_powerl2 = df_sm.iloc[:,2]
    house1_sm_powerl3 = df_sm.iloc[:,3]
    
    # Power consumption graph
    # Set common labels
    fig = plt.figure()
    fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Consumption', fontsize=20, ha='center', va='center', rotation='vertical')
    
    plt.subplots_adjust(hspace=.1) # control spacing between subplots

    axpower = plt.subplot(511)
    house1_sm_powerallphases.plot(grid=True,legend=True,label='powerallphases', color='r',linewidth=3)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(512)
    house1_sm_powerl1.plot(grid=True,legend=True,label='powerl1', color='r',linewidth=3)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(513)
    house1_sm_powerl2.plot(grid=True,legend=True,label='powerl2', color='r',linewidth=3)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(514)
    house1_sm_powerl3.plot(grid=True,legend=True,label='powerl3', color='r',linewidth=3)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    ax0 = plt.subplot(515)
    occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
    ax0.set_ylim([0,1.005])

    
    ################### count power consumption of up to block_num*resolution ###################
    # interval: resolution
    # slice num: block_num
    block_num = 80
    resolution = 50
    power_count=np.zeros([slice_num,block_num]) 
    for i in range(slice_num):
        count_dk = 0
        count_entertainment = 0
        count_lamp = 0
        count_laptop = 0
        print "i: " + str(i)
        for j in range(i*slice_interval, (i+1)*slice_interval):
            power = int(house1_sm_powerallphases[j])
            if power>=block_num*resolution:
                power=(block_num-1)*resolution
            if power<0:
                power=0
            power_count[i][int(power/resolution)] += 1
    
    
    fig = plt.figure()
    fig.text(0.5, 0.04, 'Power Consumption (' + date_string + ')', fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Count', fontsize=20, ha='center', va='center', rotation='vertical')
    plt.subplots_adjust(hspace=.5) # control spacing between subplots
    
    axcnt = plt.subplot(611)
    plt.bar(np.arange(block_num), power_count[10], 0.25)
    axcnt.set_title("No activities")
    
    axcnt = plt.subplot(612)
    plt.bar(np.arange(block_num), power_count[28], 0.25)
    axcnt.set_title("Dishwasher/Kettle + Laptop")
    
    axcnt = plt.subplot(613)
    plt.bar(np.arange(block_num), power_count[31], 0.25)
    axcnt.set_title("Laptop")
    
    axcnt = plt.subplot(614)
    plt.bar(np.arange(block_num), power_count[44], 0.25)
    axcnt.set_title("Entertainment + Laptop")
    
    axcnt = plt.subplot(615)
    plt.bar(np.arange(block_num), power_count[88], 0.25)
    axcnt.set_title("Entertainment + Lamp + Laptop")
    
    axcnt = plt.subplot(616)
    plt.bar(np.arange(block_num), power_count[95], 0.25)
    axcnt.set_title("Lamp")
    #axcnt.set_xscale('log')

#==============================================================================
#==============================================================================
# #     # Number of samplepoints
# #     N = 1200
# #     # sample spacing
# #     T = 1.0 / 800.0
# #     x = np.linspace(0.0, N*T, N)
# #     y = np.array(house1_sm_powerallphases)
# #     yf = scipy.fftpack.fft(y)
# #     xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
# #     
# #     ax_fft = plt.subplot(616)
# #     ax_fft.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# #     plt.show()
#==============================================================================
#==============================================================================
