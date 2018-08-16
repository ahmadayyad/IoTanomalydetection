# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:34:35 2018
@author: Ahmad
"""
# Imports
import pandas as pd 
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,f1_score,confusion_matrix,precision_recall_curve


#------------------Generan Setup-----------------------------------------------------

RANDOM_STATE=42
sensor_data_path='test_data_results.csv'


#Oversampling and undersampling hybrid method
sm = SMOTETomek()

#Initializing Neural Network
classifier = Sequential()



#------------------------------------------------------------------------------------


def read_csv():
    Sensor_data_csv  = pd.read_csv(sensor_data_path,sep=',')
    return Sensor_data_csv


#------------------------------------------------------------------------------------

def neural_network(start_index,end_index,apply_OS_flag):
    read_fn=read_csv()
    csv_file=read_fn.copy()
    csv_edited_X=csv_file.loc[start_index:end_index,["temp_C","humid_%"]]
    csv_edited_y=csv_file.loc[start_index:end_index,["Anomalous"]]
    
    
 # Applying standard scalar on the readings   
    csv_edited_X_scaled=csv_edited_X.copy()
    
    csv_edited_X_scaled = StandardScaler().fit_transform(csv_edited_X_scaled)

# apply_OS_flag Controls whether to apply oversampling or leave the dataset unchanged    
    if apply_OS_flag:
        X_resampled, y_resampled = sm.fit_sample(csv_edited_X_scaled, csv_edited_y.values.ravel())
    else:
        X_resampled, y_resampled=csv_edited_X_scaled,csv_edited_y.values.ravel()
        
#Counts of different classes in the set       
    unique, counts = np.unique(y_resampled, return_counts=True)
    print("counts = ",unique, counts)
    
#Split dataset to train and test.
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled.ravel(),random_state=RANDOM_STATE,test_size=0.2)
    
    
#--------------------------------Neural Network Setup--------------------------------
    
    
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(6, input_dim=2, kernel_initializer='uniform', activation='relu'))
    # Adding the second hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting our model 
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ")
    print(cm)
    
    #Recall and F1 Score
    F1_SCORE=f1_score(y_test, y_pred)
    print('F1_SCORE= ',F1_SCORE)
    
    scores = classifier.evaluate(X_resampled, y_resampled, verbose=0)
    
    print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
    # serialize model to JSON
    
    model_json = classifier.to_json()
    with open("ml_model\\classifier_seq.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("ml_model\\classifier_seq.h5")
    print("Saved model to disk")
    


if __name__ == '__main__':
    csv_main=read_csv()
    end_index=csv_main.shape[0]-1
    start_index=0
    apply_OS_flag=True
    neural_network(start_index,end_index,apply_OS_flag)
    print ('This program is being run by itself')
	
else:
	print("neural network python file being imported")