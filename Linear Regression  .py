#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[3]:


df= pd.read_csv("swedish_car_insurance - swedish_car_insurance.csv")
df.head()


# In[4]:


plt.scatter(df['X'],df['Y'],color="red")


# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis")


# In[7]:


X=df.iloc[:,:1]
X


# In[8]:


Y=df.iloc[:,1:]
Y


# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=2)


# In[10]:


LR=LinearRegression()


# In[11]:


LR.fit(X_train,Y_train)


# In[13]:


Y_pred=LR.predict(X_test)


# In[14]:


score=r2_score(Y_test,Y_pred)
score


# In[ ]:




