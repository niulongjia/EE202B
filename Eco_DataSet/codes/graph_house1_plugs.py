# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 09:41:19 2017

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

occupancy_fname_suffix="../house#01/01_occupancy_csv/"
plug_fname_suffix="../house#01/01_plugs_csv/"
df_read=pd.read_csv(occupancy_fname_suffix + "01_summer.csv")

index=10
occupancy_day = df_read.iloc[index,1:]
date_string=df_read.iloc[index,0]
date_num=convertDate(date_string)
print date_string + "  " + date_num



# Set common labels
fig = plt.figure()
fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
fig.text(0.06, 0.5, 'Power Consumption', fontsize=20, ha='center', va='center', rotation='vertical')

plt.subplots_adjust(hspace=.1) # control spacing between subplots

house1_plug1_fname=plug_fname_suffix + "01/" + date_num + ".csv"
if os.path.isfile(house1_plug1_fname)==True:
    ax1 = plt.subplot(811)
    df_house1_plug1 = pd.read_csv(house1_plug1_fname,names=['power_consumption'])
    row_house1_plug1=df_house1_plug1.iloc[:,0]
    row_house1_plug1.plot(grid=True,legend=True,label='Fridge', color='r',linewidth=1)
    ax1.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax1.set_xlabel('Time Elapsed')
#    ax1.set_ylabel('Power Consumption')


house1_plug2_fname=plug_fname_suffix + "02/" + date_num + ".csv"
if os.path.isfile(house1_plug2_fname)==True:
    ax2 = plt.subplot(812)
    df_house1_plug2 = pd.read_csv(house1_plug2_fname,names=['power_consumption'])
    row_house1_plug2=df_house1_plug2.iloc[:,0]
    row_house1_plug2.plot(grid=True,legend=True,label='Dryer', color='r',linewidth=1)
    ax2.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax2.set_xlabel('Time Elapsed')
#    ax2.set_ylabel('Power Consumption')


house1_plug3_fname=plug_fname_suffix + "03/" + date_num + ".csv"
if os.path.isfile(house1_plug3_fname)==True:
    ax3 = plt.subplot(813)
    df_house1_plug3 = pd.read_csv(house1_plug3_fname,names=['power_consumption'])
    row_house1_plug3=df_house1_plug3.iloc[:,0]
    row_house1_plug3.plot(grid=True,legend=True,label='Coffee', color='r',linewidth=1)
    ax3.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax3.set_xlabel('Time Elapsed')
#    ax3.set_ylabel('Power Consumption')


house1_plug4_fname=plug_fname_suffix + "04/" + date_num + ".csv"
if os.path.isfile(house1_plug4_fname)==True:
    ax4 = plt.subplot(814)
    df_house1_plug4 = pd.read_csv(house1_plug4_fname,names=['power_consumption'])
    row_house1_plug4=df_house1_plug4.iloc[:,0]  
    row_house1_plug4.plot(grid=True,legend=True,label='Kettle', color='r',linewidth=1)
    ax4.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax4.set_xlabel('Time Elapsed')
#    ax4.set_ylabel('Power Consumption')


house1_plug5_fname=plug_fname_suffix + "05/" + date_num + ".csv"
if os.path.isfile(house1_plug5_fname)==True:
    ax5 = plt.subplot(815)
    df_house1_plug5 = pd.read_csv(house1_plug5_fname,names=['power_consumption'])
    row_house1_plug5=df_house1_plug5.iloc[:,0]   
    row_house1_plug5.plot(grid=True,legend=True,label='Washing Machine', color='r',linewidth=1)
    ax5.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax5.set_xlabel('Time Elapsed')
#    ax5.set_ylabel('Power Consumption')


house1_plug6_fname=plug_fname_suffix + "06/" + date_num + ".csv"
if os.path.isfile(house1_plug6_fname)==True:
    ax6 = plt.subplot(816)
    df_house1_plug6 = pd.read_csv(house1_plug6_fname,names=['power_consumption'])
    row_house1_plug6=df_house1_plug6.iloc[:,0]    
    row_house1_plug6.plot(grid=True,legend=True,label='PC', color='r',linewidth=1)
    ax6.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax6.set_xlabel('Time Elapsed')
#    ax6.set_ylabel('Power Consumption')


house1_plug7_fname=plug_fname_suffix + "07/" + date_num + ".csv"
if os.path.isfile(house1_plug7_fname)==True:
    ax7 = plt.subplot(817)
    df_house1_plug7 = pd.read_csv(house1_plug7_fname,names=['power_consumption'])
    row_house1_plug7=df_house1_plug7.iloc[:,0]
    row_house1_plug7.plot(grid=True,legend=True,label='Freezer', color='r',linewidth=1)
    ax7.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
#    ax7.set_xlabel('Time Elapsed')
#    ax7.set_ylabel('Power Consumption')


ax0 = plt.subplot(818)
occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
ax0.set_ylim([0,1.005])
#ax0.set_xlabel('Time Elapsed')
#ax0.set_ylabel('Occupancy State')
#ax0.set_title(date_string+' Occupancy State')

#==============================================================================
# row1 = df_read.iloc[1,2:]
# plt.subplot(512)
# row1.plot(ylim=(0,1.005),grid=True,legend=True,title='This is title',color='r',label='This is line label',linewidth=3)
# plt.xlabel('this is xlabel')
# plt.ylabel('this is ylabel')
# 
# row2 = df_read.iloc[2,2:]
# plt.subplot(513)
# row2.plot(ylim=(0,1.005),grid=True,legend=True,title='This is title',color='r',label='This is line label',linewidth=3)
# plt.xlabel('this is xlabel')
# plt.ylabel('this is ylabel')
# 
# row3 = df_read.iloc[3,2:]
# plt.subplot(514)
# row3.plot(ylim=(0,1.005),grid=True,legend=True,title='This is title',color='r',label='This is line label',linewidth=3)
# plt.xlabel('this is xlabel')
# plt.ylabel('this is ylabel')
# 
# row4 = df_read.iloc[4,2:]
# plt.subplot(515)
# row4.plot(ylim=(0,1.005),grid=True,legend=True,title='This is title',color='r',label='This is line label',linewidth=3)
# plt.xlabel('this is xlabel')
# plt.ylabel('this is ylabel')
# plt.show()
#==============================================================================


#==============================================================================
# d = {'columns': ['T', 'G', 'C', '-', 'A', 'C', 'T', '-', 'A', 'G', 'T', 
#                  '-', 'A', 'G', 'C', '-', 'A', 'T', 'G', 'C'],
#      'data': [[97, 457, 178, 75, 718, 217, 193, 69, 184, 198,
#                777, 65, 100, 143, 477, 54, 63, 43, 55, 47]],
#      'index': [1]}
# df = pd.DataFrame(d['data'], columns=d['columns'], index=d['index'])
# df.columns.names = ['SAMPLE']
# 
# row = df.iloc[0]
# row.plot()
# plt.show()
#==============================================================================
