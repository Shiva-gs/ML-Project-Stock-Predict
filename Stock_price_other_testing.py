#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas_datareader.data as web
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

# now_time = datetime.now()
# start_time = datetime(now_time.year - 20, now_time.month , now_time.day)
# now_time
# start_time
# 

# In[8]:


now_time = datetime.now()
start_time = datetime(now_time.year - 5, now_time.month , now_time.day)
now_time
start_time


# In[9]:


stock_df = web.DataReader('AAPL','yahoo', start_time, now_time)


# In[4]:


#start_time = datetime(now_time.year - 5, now_time.month , now_time.day)

    #sp_df=web.DataReader('^GSPC','yahoo', start_time, now_time)    


# In[5]:


#sp_df.tail()


# In[10]:


stock_df.tail()


# In[11]:


# Get the Adjusted Close Price

df = stock_df[['Open','Adj Close']]
#Take a look at the new data
df.head() 

#df.to_csv('data_stocks.csv')


# In[12]:


df.tail() 


# In[20]:


# A variable for predicting 'n' days out into the future
forecast_out = 30 #'n=30' days
#Create another column (the target or dependent variable) shifted 'n' units up
#df['InitialPrediction'] = df[['Adj Close']].shift()
#df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
#print the new data set
df.head() 


# In[19]:


df.tail()


# In[21]:


### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Adj Close'],1))

#Remove the last 'n' rows
X = X[:-forecast_out]
print(X)


# In[22]:


df.head(30)


# In[23]:


df.tail()


# In[24]:


## Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(df['Adj Close'])
# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]
print(y[:-forecast_out])



# In[25]:


# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[26]:


# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)


# In[27]:


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# In[28]:


# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)


# In[29]:


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# In[30]:


# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Adj Close'],1))[-forecast_out:]
print(x_forecast) 


# # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
# #x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
# x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
# print(x_forecast[0])

# In[31]:


# Print linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
lr_pdf=pd.DataFrame(lr_prediction)
lr_pdf


# In[32]:


df.tail()


# In[71]:


df.tail()


# In[72]:


df.tail()


# In[33]:


#Error 
from sklearn.metrics import mean_squared_error,r2_score
#df['InitialPrediction'] = df[['Adj Close']].shift()
Y_orginal= np.array(df.drop(['Open'],1))
Y_true=Y_orginal[-30:]
Y_pred=lr_prediction
#mean_squared_error(Y_orginal[-30:], lr_prediction)
MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
MSE 

rmse=r2_score(Y_true, Y_pred)
rmse


# In[34]:


# Print support vector regressor model predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction) 


# In[40]:


dt=datetime.today().strftime('%Y-%m-%d')

df.tail(30) 
new_df= df.tail(30) 
new_df.head()


# In[42]:


new_df['Prediction']=lr_prediction


# In[50]:


new_df.tail()
 
stock_df.tail()


# In[251]:


# ignore last 30 days will used for prediction
X = stock_df[['High', 'Low', 'Open', 'Close', 'Volume']][:-forecast_out]
y = stock_df["Adj Close"][:-forecast_out].values.reshape(-1, 1)
print(X.shape, y.shape)


# In[252]:


# Split the data into training and testing

### BEGIN SOLUTION
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(X, y,  random_state=42)


# In[253]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScater model and fit it to the training data

### BEGIN SOLUTION
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)


# In[254]:


# Transform the training and testing data using the X_scaler and y_scaler models

### BEGIN SOLUTION
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# In[255]:


# Create a LinearRegression model and fit it to the scaled training data

### BEGIN SOLUTION
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)


# In[260]:




# predict last 30 days data
x_forecast = stock_df[['High', 'Low', 'Open', 'Close', 'Volume']][-forecast_out:]
model.fit(X_train, y_train)
lr_prediction = model.predict(x_forecast)

#scaled_df=pd.DataFrame()
#scaled_df['X_test_scaled']=X_test_scaled

# Open Price
#X_train_scaled
#lst=[]
#for i in X_train_scaled:
#    lst.append((i[3]))

#scaled_df['Adj Close']=lst

#scaled_df['Adj Close']=
#lr_prediction
scaled_df=stock_df[['Adj Close']].tail(30)


# In[266]:


#lst=[]
#for i in pred1:
#    lst.append((i[0]))
#scaled_df['Predictions']=lst
#scaled_df.tail()
scaled_df['Predictions']=lr_prediction
scaled_df.tail(10)


# In[261]:


# Used X_test_scaled, y_test_scaled, and model.predict(X_test_scaled) to calculate MSE and R2
# Make predictions using the X_test_scaled data
# Plot y_test_scaled vs y_test_scaled
# Scatter plot y_test_scaled vs predictions

### BEGIN SOLUTION
predictions = model.predict(X_test_scaled)
model.fit(X_train_scaled, y_train_scaled) 

### BEGIN SOLUTION
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test_scaled, predictions)
r2 = model.score(X_test_scaled, y_test_scaled)
### END SOLUTION

print(f"MSE: {MSE}, R2: {r2}")


# In[81]:


predict_df = stock_df


# In[210]:


len(predictions) 
X_train_scaled[1]


# In[117]:


y_test_scaled[1]
 


plt.style.use('seaborn-darkgrid')
 
plt.rc('figure', figsize=(20, 10))
# create a color palette
#plt.style.use('seaborn-whitegrid')
fig = plt.figure()

ax = plt.axes()
x = new_df.index
labels=x.strftime("%b-%d")

plt.subplot(2, 1, 1)
plt.plot(x.strftime("%b-%d"), new_df['Adj Close']);
plt.plot(x.strftime("%b-%d"), new_df['Prediction']); 
plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical')
plt.ylabel('Open vs Adj Close')

plt.subplot(2, 1, 2)
plt.plot(x.strftime("%b-%d"), scaled_df['Adj Close'].tail(30));
plt.plot(x.strftime("%b-%d"), scaled_df['Predictions'].tail(30)); 
plt.xticks(x.strftime("%b-%d"), labels, rotation='vertical')
plt.ylabel('All Paarameters')

plt.show()

plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="orange", label="Testing Data")
plt.legend()
plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
plt.title("Residual Plot")
plt.show()
