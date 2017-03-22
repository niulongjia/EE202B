# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:43:36 2017

@author: niulongjia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path

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

occupancy_fname_suffix="../house#04/04_occupancy_csv/"
plug_fname_suffix="../house#04/04_plugs_csv/"
df_read=pd.read_csv(occupancy_fname_suffix + "04_summer.csv")

index=10
occupancy_day = df_read.iloc[index,1:]
date_string=df_read.iloc[index,0]
date_num=convertDate(date_string)
print date_string + "  " + date_num


# Power consumption for the first 6 appliances
# Set common labels
fig = plt.figure()
fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
fig.text(0.06, 0.5, 'Power Consumption', fontsize=20, ha='center', va='center', rotation='vertical')

plt.subplots_adjust(hspace=.1) # control spacing between subplots

plug1_fname = plug_fname_suffix + "01/" + date_num + ".csv"
if os.path.isfile(plug1_fname)==True:
    ax1 = plt.subplot(911)
    df_house1_plug1 = pd.read_csv(plug1_fname,names=['power_consumption'])
    row_house1_plug1=df_house1_plug1.iloc[:,0]
    row_house1_plug1.plot(grid=True,legend=True,label='Fridge', color='r',linewidth=1)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax1.set_xlabel('Time Elapsed')
#    ax1.set_ylabel('Power Consumption')


plug2_fname=plug_fname_suffix + "02/" + date_num + ".csv"
if os.path.isfile(plug2_fname)==True:
    ax2 = plt.subplot(912)
    df_house1_plug2 = pd.read_csv(plug2_fname,names=['power_consumption'])
    row_house1_plug2=df_house1_plug2.iloc[:,0]
    row_house1_plug2.plot(grid=True,legend=True,label='Kitchen', color='r',linewidth=1)
    ax2.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax2.set_xlabel('Time Elapsed')
#    ax2.set_ylabel('Power Consumption')


plug3_fname=plug_fname_suffix+"03/" + date_num + ".csv"
if os.path.isfile(plug3_fname)==True:
    ax3 = plt.subplot(913)
    df_house1_plug3 = pd.read_csv(plug3_fname,names=['power_consumption'])
    row_house1_plug3=df_house1_plug3.iloc[:,0]
    row_house1_plug3.plot(grid=True,legend=True,label='Lamp', color='r',linewidth=1)
    ax3.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax3.set_xlabel('Time Elapsed')
#    ax3.set_ylabel('Power Consumption')


plug4_fname=plug_fname_suffix + "04/" + date_num + ".csv"
if os.path.isfile(plug4_fname)==True:
    ax4 = plt.subplot(914)
    df_house1_plug4 = pd.read_csv(plug4_fname,names=['power_consumption'])
    row_house1_plug4=df_house1_plug4.iloc[:,0]  
    row_house1_plug4.plot(grid=True,legend=True,label='Stereo', color='r',linewidth=1)
    ax4.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax4.set_xlabel('Time Elapsed')
#    ax4.set_ylabel('Power Consumption')


plug5_fname=plug_fname_suffix + "05/" + date_num + ".csv"
if os.path.isfile(plug5_fname)==True:
    ax5 = plt.subplot(915)
    df_house1_plug5 = pd.read_csv(plug5_fname,names=['power_consumption'])
    row_house1_plug5=df_house1_plug5.iloc[:,0]   
    row_house1_plug5.plot(grid=True,legend=True,label='Freezer', color='r',linewidth=1)
    ax5.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax5.set_xlabel('Time Elapsed')
#    ax5.set_ylabel('Power Consumption')


plug6_fname=plug_fname_suffix + "06/" + date_num + ".csv"
if os.path.isfile(plug6_fname)==True:
    ax6 = plt.subplot(916)
    df_house1_plug6 = pd.read_csv(plug6_fname,names=['power_consumption'])
    row_house1_plug6=df_house1_plug6.iloc[:,0]    
    row_house1_plug6.plot(grid=True,legend=True,label='Tablet', color='r',linewidth=1)
    ax6.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax6.set_xlabel('Time Elapsed')
#    ax6.set_ylabel('Power Consumption')

plug6_fname=plug_fname_suffix + "07/" + date_num + ".csv"
if os.path.isfile(plug6_fname)==True:
    ax6 = plt.subplot(917)
    df_house1_plug6 = pd.read_csv(plug6_fname,names=['power_consumption'])
    row_house1_plug6=df_house1_plug6.iloc[:,0]    
    row_house1_plug6.plot(grid=True,legend=True,label='Entertainment', color='r',linewidth=1)
    ax6.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax6.set_xlabel('Time Elapsed')
#    ax6.set_ylabel('Power Consumption')

plug6_fname=plug_fname_suffix + "08/" + date_num + ".csv"
if os.path.isfile(plug6_fname)==True:
    ax6 = plt.subplot(918)
    df_house1_plug6 = pd.read_csv(plug6_fname,names=['power_consumption'])
    row_house1_plug6=df_house1_plug6.iloc[:,0]    
    row_house1_plug6.plot(grid=True,legend=True,label='Microwave', color='r',linewidth=1)
    ax6.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax6.set_xlabel('Time Elapsed')
#    ax6.set_ylabel('Power Consumption')

ax0 = plt.subplot(919)
occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
ax0.set_ylim([0,1.005])
#ax0.set_xlabel('Time Elapsed')
#ax0.set_ylabel('Occupancy State')
#ax0.set_title(date_string+' Occupancy State')
