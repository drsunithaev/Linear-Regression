
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


dataset = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=[0])


# In[8]:


type(dataset)


# In[9]:


dataset.head()


# In[10]:


dataset['Temp'] = dataset['Temperature']


# In[11]:


dataset = dataset.drop('Temperature',axis=1)


# In[12]:


dataset.info()


# In[13]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[16]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()


# In[17]:


#X_train = scalar.fit_transform(X_train)
#X_test = scalar.fit_transform(X_test)


# In[18]:


from sklearn.linear_model import Ridge


# In[19]:


ridge_reg = Ridge()


# In[20]:


ridge_reg


# In[21]:


ridge_reg.fit(X_train, y_train)


# In[22]:



print(ridge_reg.coef_)


# In[23]:


print(ridge_reg.intercept_)


# In[24]:


reg_pred  = ridge_reg.predict(X_test)


# In[25]:


plt.scatter(y_test,reg_pred)


# In[26]:


residuals = y_test-reg_pred


# In[27]:


sns.displot(residuals,kind="kde")


# In[28]:


plt.scatter(reg_pred, residuals)


# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(y_test,reg_pred))


# In[30]:


print(mean_absolute_error(y_test,reg_pred))


# In[31]:


print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[32]:


from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)


# In[33]:


print(score)


# In[34]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

