import pandas as pd
import numpy as np
## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)

# Read the csv file
data=pd.read_csv("SeoulBikeData.csv",encoding= 'unicode_escape')


encode=['Seasons', 'Holiday', 'functioning_day']
for col in encode:
    dummy = pd.get_dummies(data[col], prefix = col)
    data = pd.concat([data,dummy],axis = 1)
    del data[col]
  
# Separate input & target data
inputs_df = data.drop(columns=["Rented Bike Count","Date","Unnamed: 0"],axis=1)
targets = data["Rented Bike Count"]


# Import train_test_split from sklearn library to make split of data into train sets and validation sets
from sklearn.model_selection import train_test_split
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df,targets,test_size=0.25,random_state=42)

import lightgbm 
LGBM = lightgbm.LGBMRegressor(reg_alpha= 0.729018790815961, reg_lambda= 0.009129369176067088, colsample_bytree= 0.8,
                                    subsample= 0.6, learning_rate= 0.008, max_depth= 20,num_leaves=59,min_child_samples=1,
                                    min_data_per_groups= 18,n_estimators=10000,random_state=42)
LGBM.fit(train_inputs, train_targets)

### Create a Pickle file using serialization
 
import pickle
pickle_out = open("LGBM.pkl","wb")
pickle.dump(LGBM, pickle_out)
pickle_out.close()  