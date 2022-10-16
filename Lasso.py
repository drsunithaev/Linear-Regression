
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[144]:


dataset = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv",skiprows=[0])


# In[145]:


type(dataset)


# In[146]:


dataset.head()


# In[147]:


dataset['Temp'] = dataset['Temperature']


# In[148]:


dataset = dataset.drop('Temperature',axis=1)


# In[149]:


dataset.info()


# In[150]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[151]:


from sklearn.model_selection import train_test_split


# In[152]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[153]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()


# In[154]:


#X_train = scalar.fit_transform(X_train)
#X_test = scalar.fit_transform(X_test)


# In[155]:


from sklearn.linear_model import Lasso


# In[156]:


lasso_reg = Lasso()


# In[157]:


lasso_reg


# In[158]:


lasso_reg.fit(X_train, y_train)


# In[159]:



print(lasso_reg.coef_)


# In[160]:


plt.scatter(dataset['Rain '], dataset['Temp'])


# In[161]:


print(lasso_reg.intercept_)


# In[162]:


reg_pred  = lasso_reg.predict(X_test)


# In[163]:


plt.scatter(y_test,reg_pred)


# In[164]:


residuals = y_test-reg_pred


# In[165]:


sns.displot(residuals,kind="kde")


# In[166]:


plt.scatter(reg_pred, residuals)


# In[167]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(y_test,reg_pred))


# In[168]:


print(mean_absolute_error(y_test,reg_pred))


# In[169]:


print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[170]:


from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)


# In[171]:


print(score)


# In[172]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

