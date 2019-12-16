#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[2]:


raw_data = pd.read_csv("1.04. Real-life example.csv")


# In[3]:


raw_data.head()


# In[4]:


raw_data.describe()


# In[5]:


raw_data.describe(include = "all")


# In[6]:


help(pd)


# In[7]:


data = raw_data.drop(['Model'],axis = 1)


# In[8]:


data


# In[9]:


data.isnull().sum()


# In[10]:


data_no_mv = data.dropna(axis = 0)


# In[11]:


data_no_mv.describe()


# In[12]:


sns.distplot(data_no_mv['Price'])


# In[13]:


q = data_no_mv['Price'].quantile(0.99)
req_data = data_no_mv[data_no_mv['Price']<q]


# In[14]:


req_data.describe(include = 'all')


# In[15]:


sns.distplot(req_data['Price'])


# In[16]:


q = data_no_mv['Mileage'].quantile(0.99)


# In[17]:


data_2 = req_data[req_data['Mileage']<q]


# In[18]:


data_2.describe(include = 'all')


# In[19]:


sns.distplot(data_2['Mileage'])


# In[20]:


sns.distplot(data_no_mv['Mileage'])


# In[21]:


sns.distplot(data_no_mv['EngineV'])


# In[22]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[23]:


sns.distplot(data_3['EngineV'])


# In[24]:


sns.distplot(data_no_mv['Year'])


# In[25]:


q = data_3['Year'].quantile(0.1)


# In[26]:


data_cleaned = data_3[data_3['Year']>q]


# In[27]:


data_cleaned.describe(include = 'all')


# In[28]:


sns.distplot(data_cleaned['Year'])


# In[29]:


data_cleaned = data_cleaned.reset_index(drop = True)


# In[30]:


data_cleaned


# In[31]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_Price'] = log_price


# In[32]:


data_cleaned.head()


# In[33]:


data_cleaned = data_cleaned.drop(['Price'],axis = 1)


# In[34]:


data_cleaned.head()


# In[35]:


plt.scatter(data_cleaned['Year'],data_cleaned['Log_Price'])


# In[36]:


plt.scatter(data_cleaned['Mileage'],data_cleaned['Log_Price'])


# In[37]:


plt.scatter(data_cleaned['EngineV'],data_cleaned['Log_Price'])


# In[38]:


data_cleaned.columns.values


# In[39]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]


# In[40]:


vif


# In[41]:


vif['Columns'] = variables.columns


# In[42]:


vif


# In[43]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis = 1)


# In[44]:


data_no_multicollinearity


# In[45]:


data_dummies = pd.get_dummies(data_no_multicollinearity,drop_first = True)


# In[46]:


data_dummies.head()


# ## rearrange a bit
# 

# In[47]:


data_dummies.columns.values


# In[48]:


cols = ['Mileage', 'EngineV', 'Log_Price', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[49]:


data_preprocessed = data_dummies[cols]


# In[50]:


targets = data_preprocessed['Log_Price']
inputs = data_preprocessed.drop(['Log_Price'],axis = 1)


# In[51]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)


# In[52]:


inputs_scaled = scaler.transform(inputs)


# In[53]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size = 0.2,random_state = 365)
# In[54]:


x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size = 0.2,random_state = 365)


# In[55]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[61]:


y_hat = reg.predict(x_train)


# In[62]:


plt.scatter(y_train,y_hat,alpha = 0.2)
plt.xlabel('Targets (y_train)',size  = 18)
plt.ylabel('Predictions (y_hat)',size = 18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[58]:


sns.distplot(y_train-y_hat)
plt.title('Residuals PDF',size = 18)


# In[ ]:





# In[59]:


reg.score(x_train,y_train)


# In[ ]:


y_hat_test


# In[65]:


df_performance = pd.DataFrame(np.exp(y_hat),columns = ['Prediction'])
df_performance.head()


# In[67]:


df_performance['Target'] = np.exp(y_hat)


# In[68]:


df_performance


# In[70]:


y_test = y_test.reset_index(drop = True)


# In[71]:


y_test


# In[73]:


df_performance['Residual'] = df_performance['Target'] - df_performance['Prediction']


# In[74]:


df_performance.describe()


# In[ ]:




