# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np 
import pickle

 # Loading the saved model 
loaded_model = pickle.load(open("RF_model.pkl",'rb'))


input_data = (30,10,3,2.5,250,120,35.5,195,100,16,220,95,10,4,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person has not churned')
else:
  print('The person has churned')