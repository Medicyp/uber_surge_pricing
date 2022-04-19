#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score

rides = pd.read_csv("cab_rides.csv",delimiter=',')
weather = pd.read_csv("weather.csv",delimiter=',')


# In[2]:


# print(rides)
# print(weather)


# In[3]:


# Convert the timestamp into the desirable format
rides['date_time'] = pd.to_datetime(round(rides['time_stamp']/1000,0), unit='s')
weather['date_time'] = pd.to_datetime(weather['time_stamp'], unit='s')
rides.head()




# In[4]:


# weather.head()
# print(weather)


# In[5]:


# # final_dataframe = rides.join(weather, on=['date_time'],rsuffix = '_w')
# final_dataframe = rides.merge(weather, on=['date_time'])

# #drop the null values rows

# final_dataframe=final_dataframe.dropna(axis=0)

# #make different columns of day and hour to simplify the format of date
# final_dataframe['day'] = final_dataframe.date_time.dt.dayofweek
# final_dataframe['hour'] = final_dataframe.date_time.dt.hour


# In[6]:


# print(final_dataframe)


# In[7]:


#make a column of merge date containing date merged with the location so that we can join the two dataframes on the basis of 'merge_date'
rides['merge_date'] = rides.source.astype(str) +" - "+ rides.date_time.dt.date.astype("str") +" - "+ rides.date_time.dt.hour.astype("str")
weather['merge_date'] = weather.location.astype(str) +" - "+ weather.date_time.dt.date.astype("str") +" - "+ weather.date_time.dt.hour.astype("str")



# In[8]:


# final_dataframe = rides.join(weather, on=['merge_date'],rsuffix = '_w')
final_dataframe = rides.merge(weather, on=['merge_date'])

#drop the null values rows

final_dataframe=final_dataframe.dropna(axis=0)

#make different columns of day and hour to simplify the format of date
final_dataframe['day'] = final_dataframe.date_time_y.dt.dayofweek
final_dataframe['hour'] = final_dataframe.date_time_y.dt.hour


# In[9]:


# change the index to merge_date column so joining the two datasets will not generate any error.

weather.index = weather['merge_date']

# we ignored surge value of more than 3 because the samples are very limited for surge_multiplier>3
surge_dataframe = final_dataframe[final_dataframe.surge_multiplier < 3]


# In[10]:


# print(surge_dataframe)


# In[11]:


# # filter to remove surge_multiplier of 1
# surge_dataframe.drop(surge_dataframe.index[surge_dataframe['surge_multiplier'] == 1])

# feature selection--> we are selecting the most relevant features from the dataset

x = surge_dataframe[['distance','day','hour','temp','clouds','pressure','humidity','wind','rain']]

y = surge_dataframe['surge_multiplier']


# In[12]:


print(y)
print(np.max(y))
print(type(y))


# In[13]:


# print(y)


# In[14]:


le = LabelEncoder()

#ignoring multiplier of 3 as there are only 2 values in our dataset
# le.fit([1.25,1.5,1.75,2.,2.25,2.5])
le.fit([1,1.25,1.5,1.75,2.,2.25,2.5])
y = le.transform(y)

feature_list=list(x.columns)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42) 


# In[15]:


# print(y)


# In[16]:


# Before Synthetic Minority Over-sampling TEchnique (SMOTE)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))


# In[17]:


# After SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
train_features, train_labels = sm.fit_resample(x_train, y_train)

test_features, test_labels = sm.fit_resample(x_test, y_test)


# In[18]:


# print(train_labels)


# In[19]:


# Model training
model = RandomForestClassifier(n_jobs=-1, random_state = 42, class_weight="balanced")

model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[20]:


# Feature importance
# Get numerical feature importances
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[21]:


# Evaluation of the built model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is ', round(accuracy * 100, 2), '%')


# In[22]:


# print(np.max(y_test))


# In[23]:


# print(np.sum(y_pred))

