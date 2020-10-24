#!/usr/bin/env python
# coding: utf-8

# Importing the required packages for the models

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('base.csv') #Read the datafiles and loading the datasets,trying to understand the data.
df


# In[3]:


df.shape #knowing the shape of the datasets


# In[4]:


df.dtypes #finding the datatypes of each of the columns.


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe() #understanding the datasets


# Data Visualization Process

# In[8]:


df['SO'].plot.box()


# In[9]:


df['R'].plot.box()


# In[10]:


df['AB'].plot.box()


# In[11]:


df['RA'].plot.box()


# In[12]:


df.plot(kind='box',subplots=True,layout=(6,4))


# In[13]:


sns.heatmap(df.isnull()) #finding out if their any null value graphically.


# In[14]:


dfcor=df.corr()  #finding out the correlation between each variable in the dataset.
dfcor 


# In[15]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)


# In[16]:


df['W'].plot.hist()


# In[17]:


sns.distplot(df['W'])


# In[18]:


for i in df.columns:
    plt.figure()
    sns.distplot(df[i])


# In[19]:


sns.pairplot(df)


# Trying to find out best model for the dataset

# In[20]:


from sklearn.impute import SimpleImputer


# In[21]:


imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')


# In[22]:


imputer = imputer.fit(df)


# In[23]:


df = imputer.transform(df)


# In[24]:


df


# In[25]:


df=pd.DataFrame(df)
df


# In[26]:


y=df.iloc[:,-1]


# In[27]:


y.head()


# In[28]:


y.shape


# In[29]:


x=df.iloc[:,1:-1]


# In[30]:


x.head()


# In[31]:


x.shape


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=3)


# In[33]:


x_train.shape


# In[34]:


x_test.shape


# In[35]:


y_train.shape


# In[36]:


y_test.shape


# In[37]:


lm=LinearRegression()


# In[38]:


lm.fit(x_train,y_train)


# In[39]:


lm.score(x_train,y_train)


# In[40]:


lm.coef_


# In[41]:


pred=lm.predict(x_test)


# In[42]:


print('Predicted value and actual value is',pred,y_test)


# In[46]:


from sklearn.metrics import mean_absolute_error


# In[47]:


print('Mean absolute error:',mean_absolute_error(y_test,pred))


# In[48]:


print('Root mean squared error:',np.sqrt(mean_squared_error(y_test,pred)))

Conclusion: We have used LinearRegression in this dataset and achieve the accuracy score.it is also used for predicting the number of wins for a baseball team.