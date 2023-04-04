#!/usr/bin/env python
# coding: utf-8

# # Bike Sharig Company Case Study
# 

# # MUltiple Linear Regression 

# In[1]:


#supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#import necessary libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score


# # Reading and Understanding the data 

# In[3]:


bike = pd.read_csv("day.csv")
bike.head()


# 
# # Data Inspection 

# In[4]:


bike.shape


# In[5]:


bike.info()


# In[6]:


bike.describe()


# In[7]:


# Deleting the duplicate values 
bike.drop_duplicates(inplace=True)


# In[8]:


# Again check the shape of dataframe 
bike.shape

# As the shape remains the same. It shows that no duplicate values were there in dataframe
# In[9]:


#check for the null values
bike.isnull().sum()


# #### It shows that there are no null values present in the dataframe

# In[10]:


#changing the name of some columns for better readability 
bike.rename(columns={'yr':"year",'mnth':"month",'hum':"humidity"}, inplace=True)


# In[11]:


#Converting the season colomn back to normal 
bike.season=bike.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})


# In[12]:


#Mapping the month column
bike.month=bike.month.map({1:'jan',2:'feb',3:'march',4:'april',5:'may',6:'june',7:'july',8:'aug',9:'sept',10:'oct',11:'nov',12:'dec'})


# In[13]:


# Mapping the weekdays column
bike.weekday=bike.weekday.map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})


# In[14]:


# Mapping the weathersit column
bike.weathersit=bike.weathersit.map({1:'Clear',2:'Misty',3:'Light_snowrain',4:'Heavy_snowrain'})


# #### Lets check the head of dataframe now to see the result of mapping

# In[15]:


bike.head()


# ## Dropping the unwanted columns
# ### Instant is nothing but  only the index for the row
# ### dteday has date which can be seen by year and month column
# ###  We also don't need casual and registered column as their combined values are present in cnt column.

# In[16]:


bike.drop(['instant', 'dteday', 'casual', 'registered'],axis=1, inplace=True)


# In[17]:


bike.head()


# # Exploratory Data Analysis

# ### Visualizing the Categorical variable

# In[18]:


plt.figure(figsize=(22, 12))
plt.subplot(2,4,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike)
plt.subplot(2,4,2)
sns.boxplot(x = 'month', y = 'cnt', data = bike)
plt.xticks(rotation=45)
plt.subplot(2,4,3)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike)
plt.subplot(2,4,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike)
plt.subplot(2,4,5)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike)
plt.subplot(2,4,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike)
plt.subplot(2,4,7)
sns.boxplot(x = 'year', y = 'cnt', data = bike)
plt.show()


# #### Creating  barplot for categorical variables

# In[19]:



def plot_cat_columns(column):
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    sns.barplot(column,'cnt',data=bike)
    plt.xticks(rotation=45)
    plt.subplot(1,2,2)
    sns.barplot(column,'cnt',data=bike, hue='year',palette='Set1')
    plt.legend(labels=['2018', '2019'])
    plt.xticks(rotation=45)
    plt.show()


# In[20]:


# PLoting Bargraph for season column
plot_cat_columns('season')


#  Fall season has the most no of bookings. And bookings have increased drastically in the year 2019 in all the seasons.

# In[21]:


# Ploting bargraph for month column
plot_cat_columns('month')


# Months with highest no. of bookings are may, june, july, aug, sep and oct.Bookings increased from starting of the year till september and then it started decreasing as we approached the end of year.Where as the number of bookings for each month incresed in 2019 as compared to 2018

# In[22]:


# Ploting bargraph for weekday column
plot_cat_columns('weekday')


# Thursday, friday, saturday and sunday have more no of bookings as compared to starting of the week. 

# In[23]:


# Ploting bargraph for weathersit column
plot_cat_columns('weathersit')


# Most number of bookings are on clear weather days. And in comparison to previous year, i.e 2018, booking increased for each weather situation in 2019.

# In[24]:


# Ploting bargraph for holiday column
plot_cat_columns('holiday')


# bookings seems to be low on holidays. 

# In[25]:


# plotting barghaph for workingday column

plot_cat_columns('workingday')


# Booking seemed to be fairly same on both the  working as well as on non-working days. But the count increased in year 2019.

# In[26]:


# plotting barghaph for year column

plot_cat_columns('year')


# Year 2019 has more bookings as compared to 2018. Which shows a growth in the business.

# In[27]:


# Visualizing the numerical columns

sns.pairplot(data=bike,vars=['temp','atemp','humidity','windspeed','cnt'])
plt.show()


# In[28]:


# Checking the correlation between the numerical variables

plt.figure(figsize = (8,8))
matrix = np.triu(bike[['temp','atemp','humidity','windspeed','cnt']].corr())
sns.heatmap(bike[['temp','atemp','humidity','windspeed','cnt']].corr(), annot = True, cmap="flare", mask=matrix)
plt.title("Correlation between Numerical Variables")
plt.show()


# There is a linear relationship between temp and atemp. Both of the parameters cannot be used in the model due to multicolinearity. So we will drop one out of them.

# # Data Preparation

# In[29]:


# Creating dummy variables for columns Season, month, weekday,weathersit


# In[30]:


season_dumy= pd.get_dummies(bike.season, drop_first=True)


# In[31]:


month_dumy= pd.get_dummies(bike.month, drop_first= True)


# In[32]:


weekday_dumy= pd.get_dummies(bike.weekday, drop_first= True)


# In[33]:


weather_dumy= pd.get_dummies(bike.weathersit, drop_first= True)


# In[ ]:





# In[34]:


## Concating the dummy dataframe with original
bike= pd.concat([bike, season_dumy, month_dumy, weekday_dumy, weather_dumy], axis=1)


# In[35]:


bike.head()


# In[36]:


# Droping unnecessary columns
bike= bike.drop(['season', 'month', 'weekday','weathersit'], axis = 1)


# In[37]:


bike.head()


# In[38]:


bike.info()


# In[39]:


#shape of new dataframe
bike.shape


# # Splitting the Data into Train and Test Sets 

# In[40]:


# splitting the dataframe into Train and Test

np.random.seed(0)
df_train, df_test = train_test_split(bike, train_size = 0.7, random_state = 100)


# In[41]:


# check the shape of training datatset

df_train.shape


# In[42]:


# check the shape of testing datatset

df_test.shape


# In[43]:


# Using MinMaxScaler to Rescaling the features

scaler = MinMaxScaler()


# In[44]:


# verifying the head of dataset before scaling.

df_train.head()


# In[45]:


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['temp','atemp','humidity','windspeed','cnt']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[46]:


# verifying the head after appying scaling.

df_train.head()


# In[47]:


# describing the dataset

df_train.describe()


# In[48]:


# check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (25,25))
matrix = np.triu(df_train.corr())
sns.heatmap(df_train.corr(), annot = True, cmap="RdYlGn", mask=matrix)
plt.show()


# cnt seems to have correlation with year variable and temp. Similarly, Misty and humidity show correlation. Spring season with Jan and Feb month, Summer season with may month and Winter season with oct and nov month show good correlation.

# In[49]:


# Visualizing one of the correlation to see the trends via Scatter plot.

plt.figure(figsize=[6,6])
plt.scatter(df_train.temp, df_train.cnt)
plt.show()


# Visualization confirms the positive correlation between temp and cnt.

# In[50]:


# Building the Linear Model

y_train = df_train.pop('cnt')
X_train = df_train


# In[51]:


# Recursive feature elimination 

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[52]:


#List of variables selected in top 15 list

list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[53]:


# selecting the selected variable via RFE in col list

col = X_train.columns[rfe.support_]
print(col)


# In[54]:


# checking which columns has been rejected

X_train.columns[~rfe.support_]


# In[55]:


# Generic function to calculate VIF of variables

def calculateVIF(df):
    vif = pd.DataFrame()
    vif['Features'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif 


# In[56]:


# dataframe with RFE selected variables

X_train_rfe = X_train[col]


# In[57]:


# calculate VIF

calculateVIF(X_train_rfe)


# humidity shows high VIF value.

# # Building a Linear Model

# In[58]:


# Building 1st linear regression model

X_train_lm_1 = sm.add_constant(X_train_rfe)
lr_1 = sm.OLS(y_train,X_train_lm_1).fit()
print(lr_1.summary())


# In[59]:


# As humidity shows high VIF values hence we can drop it
X_train_new = X_train_rfe.drop(['humidity'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# VIF values seems to be good now. But lets see if we can reduce it further 

# In[60]:


# Building 2nd linear regression model

X_train_lm_2 = sm.add_constant(X_train_new)
lr_2 = sm.OLS(y_train,X_train_lm_2).fit()
print(lr_2.summary())


# In[61]:


# We can drop nov variable as it has high p-value
X_train_new = X_train_new.drop(['nov'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# VIF doesnt seems to be affected much even after dropping few of the variables. Lets proceed more to observe better model.

# In[62]:


# Building 3rd linear regression model

X_train_lm_3 = sm.add_constant(X_train_new)
lr_3 = sm.OLS(y_train,X_train_lm_3).fit()
print(lr_3.summary())


# In[63]:


# We can drop dec variable as it has high p-value
X_train_new = X_train_new.drop(['dec'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[64]:


# Building 4th linear regression model

X_train_lm_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_lm_4).fit()
print(lr_4.summary())


# In[65]:


# We can drop jan variable as it has high p-value
X_train_new = X_train_new.drop(['jan'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[66]:


# Building 5th linear regression model

X_train_lm_5 = sm.add_constant(X_train_new)
lr_5 = sm.OLS(y_train,X_train_lm_5).fit()
print(lr_5.summary())


# In[67]:


# We can drop july variable as it has high p-value
X_train_new = X_train_new.drop(['july'], axis = 1)

# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# VIF value now seems to be good as it came below 5.

# In[68]:


# Building 6th linear regression model

X_train_lm_6 = sm.add_constant(X_train_new)
lr_6 = sm.OLS(y_train,X_train_lm_6).fit()
print(lr_6.summary())


# We can cosider the above model i.e lr_6, as it seems to have very low multicolinearity between the predictors and the p-values for all the predictors seems to be significant.
# F-Statistics value of 248.4 (which is greater than 1) and the p-value of 1.47e-186 i.e almost equals to zero, states that the overall model is significant

# In[69]:


# Checking the parameters and their coefficient values
lr_6.params


# #  Residual Analysis of the train data and validation

# In[70]:


X_train_lm_6


# In[71]:


y_train_pred = lr_6.predict(X_train_lm_6)


# ### Normality of error terms

# In[72]:


# Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20) 
plt.xlabel('Errors', fontsize = 18)


# Error terms are showing normal distribution

# ### Multi Colinearity

# In[73]:


calculateVIF(X_train_new)


# In[74]:


plt.figure(figsize=(15,8))
sns.heatmap(X_train_new.corr(),annot = True, cmap="RdYlGn")
plt.show()


# VIF values are less than 5 which is good and also there is no multicolinearity as seen from the heatmap.

# #### Linearity

# In[75]:


# Linear relationship validation using CCPR plot
# Component and component plus residual plot

sm.graphics.plot_ccpr(lr_6, 'temp')
plt.show()

sm.graphics.plot_ccpr(lr_6, 'sept')
plt.show()

sm.graphics.plot_ccpr(lr_6, 'windspeed')
plt.show()


# We can see linearity from the above visualizations.

# ### Homoscedasticity

# In[76]:


y_train_pred = lr_6.predict(X_train_lm_6)
residual = y_train - y_train_pred
sns.scatterplot(y_train,residual)
plt.plot(y_train,(y_train - y_train), '-r')
plt.xlabel('Count')
plt.ylabel('Residual')
plt.show()


# No visible pattern observed from above plot for residuals.
# 
# ## Independence of residuals
# Durbin-Watson value of final model lr_6 is 2.085, which signifies there is no autocorrelation.

# # Making Predictions Using the Final Model

# In[77]:


# Applying scaling on the test dataset

num_vars = ['temp', 'atemp', 'humidity', 'windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()


# In[78]:


df_test.describe()


# In[79]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[80]:


col1 = X_train_new.columns

X_test = X_test[col1]

# Adding constant variable to test dataframe
X_test_lm_6 = sm.add_constant(X_test)


# In[81]:


y_pred = lr_6.predict(X_test_lm_6)


# In[82]:


# r2 = r2_score(y_test, y_pred)
round(r2,4)


# # Model Evaluation

# ### Let's now plot the graph for actual versus predicted values

# In[ ]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20) 
plt.xlabel('y_test', fontsize = 18)
plt.ylabel('y_pred', fontsize = 16) 


# In[ ]:


round(lr_6.params,4)


# #### We can see that the equation of our best fitted line is:
#  

# #### cnt = 0.1909 + 0.2341 * year - 0.0963 * holiday + 0.4777 * temp - 0.1481 * windspeed + 0.0910 * sep - 0.2850 * Light_snowrain -   0.0787 * Misty - 0.0554 * spring + 0.0621 * summer + 0.0945 * winter
# 
# 

# In[ ]:


# Calculating Adjusted-R^2 value for the test dataset

adjusted_r2 = round(1-(1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1),4)
print(adjusted_r2)


# In[ ]:


# Visualizing the fit on the test data
# plotting a Regression plot

plt.figure()
sns.regplot(x=y_test, y=y_pred, ci=68, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
plt.show()


# # Comparision between Training and Testing dataset:

# - Train dataset R^2          : 0.833
# - Test dataset R^2           : 0.8038
# - Train dataset Adjusted R^2 : 0.829    
# - Test dataset Adjusted R^2  : 0.7944

# ## FINAL REPORT
# 
# ### Demand of bikes depend on year, holiday, temp, windspeed, sep, Light_snowrain, Misty, spring, summer and winter.

# 
# 
# As per our final Model, the top 3 predictor variables that influences the bike booking are:
# 
# #### Temperature (temp) - A coefficient value of ‘0.4777’ indicated that a unit increase in temp variable increases the bike hire numbers by 0.4777 units.
# 
# #### Year (yr) - A coefficient value of ‘0.2341’ indicated that a unit increase in yr variable increases the bike hire numbers by 0.2341 units.
# 
# ### So, it's suggested to consider these variables utmost importance while planning, to achive maximum Booking
# ### The next best features that can also be considered are
# 
# #### Light_snowrain: - A coefficient value of ‘0.2850’ indicated that a unit increase in Light_snowrain variable increases the bike hire numbers by 0.2850 units.
# 

# In[ ]:




