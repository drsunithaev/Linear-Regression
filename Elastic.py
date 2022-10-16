
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


dataset = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=[0])


# In[65]:


type(dataset)


# In[66]:


dataset.head()


# In[67]:


dataset['Temp'] = dataset['Temperature']


# In[68]:


dataset = dataset.drop('Temperature',axis=1)


# In[69]:


dataset.info()


# In[70]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[73]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()


# In[74]:


#X_train = scalar.fit_transform(X_train)
#X_test = scalar.fit_transform(X_test)


# In[75]:


from sklearn.linear_model import ElasticNet


# In[76]:


elastic_reg = ElasticNet()


# In[77]:


elastic_reg


# In[78]:


elastic_reg.fit(X_train, y_train)


# In[79]:



print(elastic_reg.coef_)


# In[80]:


print(elastic_reg.intercept_)


# In[81]:


reg_pred  = elastic_reg.predict(X_test)


# In[82]:


plt.scatter(y_test,reg_pred)


# In[83]:


residuals = y_test-reg_pred


# In[84]:


sns.displot(residuals,kind="kde")


# In[85]:


plt.scatter(reg_pred, residuals)


# In[86]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(y_test,reg_pred))


# In[87]:


print(mean_absolute_error(y_test,reg_pred))


# In[88]:


print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[89]:


from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)


# In[90]:


print(score)


# In[91]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

