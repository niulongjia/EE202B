# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:51:09 2017

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
df_read=pd.read_csv(occupancy_fname_suffix + "01_summer.csv")

index=10
occupancy_day = df_read.iloc[index,1:]
date_string=df_read.iloc[index,0]
date_num=convertDate(date_string)
print date_string + "  " + date_num






house1_sm_fname="../house#01/01_sm_csv/" + date_num + ".csv"
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
    house1_sm_currentneutral = df_sm.iloc[:,4]
    house1_sm_currentl1 = df_sm.iloc[:,5]
    house1_sm_currentl2 = df_sm.iloc[:,6]
    house1_sm_currentl3 = df_sm.iloc[:,7]
    house1_sm_voltagel1 = df_sm.iloc[:,8]
    house1_sm_voltagel2 = df_sm.iloc[:,9]
    house1_sm_voltagel3 = df_sm.iloc[:,10]
    house1_sm_phaseanglevoltagel2l1 = df_sm.iloc[:,11]
    house1_sm_phaseanglevoltagel3l1 = df_sm.iloc[:,12]
    house1_sm_phaseanglecurrentvoltagel1 = df_sm.iloc[:,13]
    house1_sm_phaseanglecurrentvoltagel2 = df_sm.iloc[:,14]
    house1_sm_phaseanglecurrentvoltagel3 = df_sm.iloc[:,15]
    
    # Power consumption graph
    # Set common labels
    fig = plt.figure()
    fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Power Consumption', fontsize=20, ha='center', va='center', rotation='vertical')
    
    plt.subplots_adjust(hspace=.1) # control spacing between subplots

    axpower = plt.subplot(511)
    house1_sm_powerallphases.plot(grid=True,legend=True,label='powerallphases', color='r',linewidth=1)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(512)
    house1_sm_powerl1.plot(grid=True,legend=True,label='powerl1', color='r',linewidth=1)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(513)
    house1_sm_powerl2.plot(grid=True,legend=True,label='powerl2', color='r',linewidth=1)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axpower = plt.subplot(514)
    house1_sm_powerl3.plot(grid=True,legend=True,label='powerl3', color='r',linewidth=1)
    axpower.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    ax0 = plt.subplot(515)
    occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
    ax0.set_ylim([0,1.005])
    
    
    # Voltage/Current consumption graph
    # Set common labels
    fig = plt.figure()
    fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Voltage/Current Consumption', fontsize=20, ha='center', va='center', rotation='vertical')
    
    plt.subplots_adjust(hspace=.1) # control spacing between subplots

    axVI = plt.subplot(811)
    house1_sm_currentneutral.plot(grid=True,legend=True,label='currentneutral', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(812)
    house1_sm_currentl1.plot(grid=True,legend=True,label='currentl1', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(813)
    house1_sm_currentl2.plot(grid=True,legend=True,label='currentl2', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(814)
    house1_sm_currentl3.plot(grid=True,legend=True,label='currentl3', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(815)
    house1_sm_voltagel1.plot(grid=True,legend=True,label='voltagel1', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(816)
    house1_sm_voltagel2.plot(grid=True,legend=True,label='voltagel2', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axVI = plt.subplot(817)
    house1_sm_voltagel3.plot(grid=True,legend=True,label='voltagel3', color='r',linewidth=1)
    axVI.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    ax0 = plt.subplot(818)
    occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
    ax0.set_ylim([0,1.005])
    
    # Phase graph
    # Set common labels
    fig = plt.figure()
    fig.text(0.5, 0.04, 'Time Elapsed (' + date_string + ')', fontsize=20, ha='center', va='center')
    fig.text(0.06, 0.5, 'Voltage Phase', fontsize=20, ha='center', va='center', rotation='vertical')
    
    plt.subplots_adjust(hspace=.1) # control spacing between subplots

    axphase = plt.subplot(611)
    house1_sm_phaseanglevoltagel2l1.plot(grid=True,legend=True,label='phaseanglevoltagel2l1', color='r',linewidth=1)
    axphase.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axphase = plt.subplot(612)
    house1_sm_phaseanglevoltagel3l1.plot(grid=True,legend=True,label='phaseanglevoltagel3l1', color='r',linewidth=1)
    axphase.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axphase = plt.subplot(613)
    house1_sm_phaseanglecurrentvoltagel1.plot(grid=True,legend=True,label='phaseanglecurrentvoltagel1', color='r',linewidth=1)
    axphase.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axphase = plt.subplot(614)
    house1_sm_phaseanglecurrentvoltagel2.plot(grid=True,legend=True,label='phaseanglecurrentvoltagel2', color='r',linewidth=1)
    axphase.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
    
    axphase = plt.subplot(615)
    house1_sm_phaseanglecurrentvoltagel3.plot(grid=True,legend=True,label='phaseanglecurrentvoltagel3', color='r',linewidth=1)
    axphase.xaxis.set_ticklabels([]) # hide tick text while keeping grid lines
      
    ax0 = plt.subplot(616)
    occupancy_day.plot(grid=True,legend=True,label='Occupancy',color='b',linewidth=3)
    ax0.set_ylim([0,1.005])
