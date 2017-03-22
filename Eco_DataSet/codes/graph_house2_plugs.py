# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:39:23 2017

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

occupancy_fname_suffix="../house#02/02_occupancy_csv/"
plug_fname_suffix="../house#02/02_plugs_csv/"
df_read=pd.read_csv(occupancy_fname_suffix + "02_summer.csv")

index=18
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

house1_plug1_fname= plug_fname_suffix + "01/" + date_num + ".csv"
if os.path.isfile(house1_plug1_fname)==True:
    ax1 = plt.subplot(711)
    df_house1_plug1 = pd.read_csv(house1_plug1_fname,names=['power_consumption'])
    row_house1_plug1=df_house1_plug1.iloc[:,0]
    row_house1_plug1.plot(grid=True,legend=True,label='Tablet', color='r',linewidth=1)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax1.set_xlabel('Time Elapsed')
#    ax1.set_ylabel('Power Consumption')


house1_plug2_fname= plug_fname_suffix + "02/" + date_num + ".csv"
if os.path.isfile(house1_plug2_fname)==True:
    ax2 = plt.subplot(712)
    df_house1_plug2 = pd.read_csv(house1_plug2_fname,names=['power_consumption'])
    row_house1_plug2=df_house1_plug2.iloc[:,0]
    row_house1_plug2.plot(grid=True,legend=True,label='Dishwasher', color='r',linewidth=1)
    ax2.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax2.set_xlabel('Time Elapsed')
#    ax2.set_ylabel('Power Consumption')


house1_plug3_fname= plug_fname_suffix + "03/" + date_num + ".csv"
if os.path.isfile(house1_plug3_fname)==True:
    ax3 = plt.subplot(713)
    df_house1_plug3 = pd.read_csv(house1_plug3_fname,names=['power_consumption'])
    row_house1_plug3=df_house1_plug3.iloc[:,0]
    row_house1_plug3.plot(grid=True,legend=True,label='Air exhaust', color='r',linewidth=1)
    ax3.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax3.set_xlabel('Time Elapsed')
#    ax3.set_ylabel('Power Consumption')


house1_plug4_fname= plug_fname_suffix + "04/" + date_num + ".csv"
if os.path.isfile(house1_plug4_fname)==True:
    ax4 = plt.subplot(714)
    df_house1_plug4 = pd.read_csv(house1_plug4_fname,names=['power_consumption'])
    row_house1_plug4=df_house1_plug4.iloc[:,0]  
    row_house1_plug4.plot(grid=True,legend=True,label='Fridge', color='r',linewidth=1)
    ax4.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax4.set_xlabel('Time Elapsed')
#    ax4.set_ylabel('Power Consumption')


house1_plug5_fname= plug_fname_suffix + "05/" + date_num + ".csv"
if os.path.isfile(house1_plug5_fname)==True:
    ax5 = plt.subplot(715)
    df_house1_plug5 = pd.read_csv(house1_plug5_fname,names=['power_consumption'])
    row_house1_plug5=df_house1_plug5.iloc[:,0]   
    row_house1_plug5.plot(grid=True,legend=True,label='Entertainment', color='r',linewidth=1)
    ax5.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax5.set_xlabel('Time Elapsed')
#    ax5.set_ylabel('Power Consumption')


house1_plug6_fname= plug_fname_suffix + "06/" + date_num + ".csv"
if os.path.isfile(house1_plug6_fname)==True:
    ax6 = plt.subplot(716)
    df_house1_plug6 = pd.read_csv(house1_plug6_fname,names=['power_consumption'])
    row_house1_plug6=df_house1_plug6.iloc[:,0]    
    row_house1_plug6.plot(grid=True,legend=True,label='Freezer', color='r',linewidth=1)
    ax6.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax6.set_xlabel('Time Elapsed')
#    ax6.set_ylabel('Power Consumption')



ax0 = plt.subplot(717)
occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
ax0.set_ylim([0,1.005])
#ax0.set_xlabel('Time Elapsed')
#ax0.set_ylabel('Occupancy State')
#ax0.set_title(date_string+' Occupancy State')


# Power consumption for the rest of 6 appliances
# Set common labels
fig = plt.figure()
fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
fig.text(0.06, 0.5, 'Power Consumption', fontsize=20, ha='center', va='center', rotation='vertical')

house1_plug7_fname= plug_fname_suffix + "07/" + date_num + ".csv"
if os.path.isfile(house1_plug7_fname)==True:
    ax7 = plt.subplot(711)
    df_house1_plug7 = pd.read_csv(house1_plug7_fname,names=['power_consumption'])
    row_house1_plug7=df_house1_plug7.iloc[:,0]
    row_house1_plug7.plot(grid=True,legend=True,label='Kettle', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')

house1_plug8_fname= plug_fname_suffix + "08/" + date_num + ".csv"
if os.path.isfile(house1_plug8_fname)==True:
    ax7 = plt.subplot(712)
    df_house1_plug8 = pd.read_csv(house1_plug8_fname,names=['power_consumption'])
    row_house1_plug8=df_house1_plug8.iloc[:,0]
    row_house1_plug8.plot(grid=True,legend=True,label='Lamp', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')

house1_plug9_fname= plug_fname_suffix + "09/" + date_num + ".csv"
if os.path.isfile(house1_plug9_fname)==True:
    ax7 = plt.subplot(713)
    df_house1_plug9 = pd.read_csv(house1_plug9_fname,names=['power_consumption'])
    row_house1_plug9=df_house1_plug9.iloc[:,0]
    row_house1_plug9.plot(grid=True,legend=True,label='Laptops', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')

house1_plug10_fname= plug_fname_suffix + "10/" + date_num + ".csv"
if os.path.isfile(house1_plug10_fname)==True:
    ax7 = plt.subplot(714)
    df_house1_plug10 = pd.read_csv(house1_plug10_fname,names=['power_consumption'])
    row_house1_plug10=df_house1_plug10.iloc[:,0]
    row_house1_plug10.plot(grid=True,legend=True,label='Stove', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')

house1_plug11_fname= plug_fname_suffix + "11/" + date_num + ".csv"
if os.path.isfile(house1_plug11_fname)==True:
    ax7 = plt.subplot(715)
    df_house1_plug11 = pd.read_csv(house1_plug11_fname,names=['power_consumption'])
    row_house1_plug11=df_house1_plug11.iloc[:,0]
    row_house1_plug11.plot(grid=True,legend=True,label='TV', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')

house1_plug12_fname= plug_fname_suffix + "12/" + date_num + ".csv"
if os.path.isfile(house1_plug12_fname)==True:
    ax7 = plt.subplot(716)
    df_house1_plug12 = pd.read_csv(house1_plug12_fname,names=['power_consumption'])
    row_house1_plug12=df_house1_plug12.iloc[:,0]
    row_house1_plug12.plot(grid=True,legend=True,label='Stereo', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')
    
ax0 = plt.subplot(717)
occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
ax0.set_ylim([0,1.005])
#ax0.set_xlabel('Time Elapsed')
#ax0.set_ylabel('Occupancy State')
#ax0.set_title(date_string+' Occupancy State')