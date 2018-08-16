# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Qt4Agg',warn=False, force=True)
import logging

#-----------------------------------------------------
sensor_data_path='test_data_results.csv'
fig = plt.figure(figsize=(6,6), dpi=100)
ax1 = fig.add_subplot(111)
#-----------setup for indexing------------------------
no_of_hours=6
sending_freq_per_min=2
read_index=-500
end_index=0
no_of_nodes=5
window_size =no_of_nodes*sending_freq_per_min*60*no_of_hours
prev_file_length=0
#-----------------------------------------------------


# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

#-----------------------------------------------------
def read_sensor_data():
    try:
        Sensor_data_csv  = pd.read_csv(sensor_data_path,sep=',')
    except IOError:
        logging.exception('')
    if Sensor_data_csv.empty:
        raise ValueError('No data available')
    return Sensor_data_csv

#-----------------------------------------------------
def modify_file(csv_file):
    global read_index,end_index,prev_file_length,skip_flag
    #=================================================
    file_length=csv_file.shape[0]
    skip_flag=False
    #=================================================
    if file_length == prev_file_length and end_index == file_length :
        skip_flag=True
        return csv_file
    else:
        prev_file_length=file_length
        if file_length < window_size:
            end_index = file_length
            return csv_file
        
        else:
            if file_length>end_index+no_of_nodes:
                read_index=read_index+500
            else:
                read_index=read_index+(file_length-end_index)
                
            end_index=read_index+window_size-1
            print("Start_Index= ",read_index) 
            print("End_Index= ",end_index)
            return csv_file.loc[read_index:end_index,:].copy()
        
        

def animate(i):
    results_file=read_sensor_data()
    results_file_copy=modify_file(results_file)
    results_file_copy=results_file_copy.reset_index(drop=True)
    ax1.clear()
    new_train = results_file_copy.loc[results_file_copy['Anomalous'] == 0]
    anamolies = results_file_copy.loc[results_file_copy['Anomalous'] == 1]
    # Final plot
    ax1.scatter(new_train["temp_C"],new_train["humid_%"] , s=10, c='b', marker="o", label='Normal')
    ax1.scatter(anamolies["temp_C"],anamolies["humid_%"], s=10, c='r', marker="o", label='Anamoly')
    plt.legend(loc='upper left');
    plt.xlabel("Temp")
    plt.ylabel("Humidity")
#    print(anamolies)
    
        
    

def main_fn():
    global fig,ax1,ani
    ani=animation.FuncAnimation(fig,animate,interval=1000,save_count=100,repeat=True,blit=False)
    #ani.save('lines.mp4', writer=writer)
    plt.show()



if __name__ == "__main__":
    main_fn()
