# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:05:01 2018

@author: Ahmad
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 21:10:41 2018

@author: Ahmad
"""
# imports
import DBSCAN_FN 
import pandas as pd
import numpy as np 
import time




# intialization of dataframes
sleep_period=1 #sleep period in seconds
skip_flag=False


#-----------setup for indexing and window size------------------------
no_of_hours=5
sending_freq_per_min=2
read_index=-500
shift_value_max=500
end_index=0
no_of_nodes=5
window_size = no_of_nodes*sending_freq_per_min*60*no_of_hours
prev_file_length=0
sensor_data_path='file_with_anamolies.csv'
#-----------------------------------------------------
full_file = pd.DataFrame(columns=["timestamp","id","temp_C","humid_%","iaq","Anomalous"])
anam_table_full =pd.DataFrame(columns=["temp_C","humid_%","Anomalous"])
#-----------------------------------------------------
full_file.to_csv("full_results.csv",index=False)
#-----------------------------------------------------
# function to read the csv file
def read_sensor_data():
    try:
        Sensor_data_csv  = pd.read_csv(sensor_data_path,sep=',')
    except IOError:
        print('error happened while opening the file')
    if Sensor_data_csv.empty:
        raise ValueError('No data available')
    return Sensor_data_csv
#-----------------------------------------------------
    

#-----------------------------------------------------
def modify_file(csv_file):
    global read_index,end_index,prev_file_length,skip_flag,shift_value
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
            end_index = file_length-1        
            read_index=0
        
        else:
            if file_length>end_index+shift_value_max:
                shift_value=shift_value_max
                read_index=read_index+shift_value
            else:
                shift_value=file_length-end_index
                read_index=read_index+shift_value
                
            end_index=read_index+window_size-1
#            print("Start_Index= ",read_index) 
#            print("End_Index= ",end_index)
#            print("Shift value= ",shift_value)
        return csv_file.loc[read_index:end_index,["timestamp","id","temp_C","humid_%","iaq"]]
        
        
#---------------main function---------------------
def main_fn():
    global anam_table_full,full_file
        
    while True:


#reading the CSV file and preproccessing it
        Sensor_data=read_sensor_data()
    
    
#modify the CSV file and return the indexed file
        test_file_indexed=modify_file(Sensor_data)
    
        test_file_indexed_copy=test_file_indexed.copy()
        test_file_indexed_copy=test_file_indexed_copy.reset_index(drop=True)
    
    
#   test for skip flag
        if skip_flag:
            break  
    
#   print(test_file_indexed)
        file_for_LOF=test_file_indexed_copy.loc[:,["temp_C","humid_%"]]
    
    
#   Apply DBSCAN to the data chunk   
        scanned_file,anam_table_short=DBSCAN_FN.apply_dbscan(file_for_LOF)
   
    
        if full_file.shape[0]< window_size:
            append_df=test_file_indexed_copy.copy()
            append_df=append_df.assign(Anomalous=scanned_file.loc[:,"Anomalous"])
        else:
            append_df=test_file_indexed_copy.tail(shift_value).copy()
            Anomalous_flag=scanned_file.loc[:,"Anomalous"]
            Anomalous_flag_append=Anomalous_flag.tail(shift_value)
            append_df=append_df.assign(Anomalous=Anomalous_flag_append)
    
    
        anam_table_full=anam_table_full.append(anam_table_short,ignore_index = True)
        anam_table_full=anam_table_full.drop_duplicates()
        anam_table_full=anam_table_full.reset_index(drop=True)

#       Add the data to the full file
 
        full_file=full_file.append(append_df,ignore_index = True)
        full_file = full_file.reset_index(drop=True)
#        print("full file size= " ,full_file.shape[0])

        with open('full_results.csv', 'a') as f:
            append_df.to_csv(f, header=False,index=False)  

    

  

#       timer to sleep for x sec
#       time.sleep(sleep_period)
    
    
    
    
    print("Total no of Anomalies= ",anam_table_full.shape[0])      
    print("Final Full File size= ",full_file.shape[0])
    total_no_of_Anomalies=full_file.loc[full_file["Anomalous"]==1].shape[0]
    print("Anomalies from main file = ",total_no_of_Anomalies)
 
    
if __name__ == "__main__":
    main_fn()
    
    