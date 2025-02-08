#**Problem Statement**

- **Description:** Develop a regression model to predict future sales based
on historical data. This helps in forecasting and planning for inventory,
marketing, and budgeting.
- **Why:** Accurate sales predictions enable better decision-making and
resource allocation.
- **Tasks:**

    ▪ Gather historical sales data.

    ▪ Preprocess data (handling missing values, encoding categorical
    variables).

    ▪ Example datasets Click Here

    ▪ Train regression models (e.g., linear regression, random forest).

    ▪ Evaluate model performance and make predictions.

**Using Google Colab**

from google.colab import files
uploaded = files.upload()


**Import Necessary Libraries**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

**Load the dataset**

BigMart = pd.read_csv('BigMart Sales Data.csv')

**First five count from dataset**

BigMart.head()

**Check the data type of the columns**

BigMart.info()

**Checking the null values**

BigMart.isnull().sum().sum()

BigMart.isnull().sum()

As the count of null values are high so we are filling it with `mean` value and for categorical column filling with `mode` value

BigMart['Item_Weight'].fillna(BigMart['Item_Weight'].mean(),inplace = True)

BigMart['Outlet_Size'].fillna(BigMart['Outlet_Size'].mode()[0],inplace = True)

BigMart.isnull().sum()

**Checking the duplicate value**

BigMart.duplicated().sum()

**Checking the outliers**

for i in BigMart.columns:
  if BigMart[i].dtype!= 'object':
    plt.boxplot(BigMart[i])
    plt.title(i)
    plt.show()

As outlier is present for 2 columns so we are treating them with `IQR method`

# IQR method
out_cols = ['Item_Visibility', 'Item_Outlet_Sales']

for i in out_cols:
  Q1 = BigMart[i].quantile(0.25)
  Q3 = BigMart[i].quantile(0.75)
  IQR = Q3 - Q1
  upper_limit = Q3+1.5*IQR
  lower_limit = Q1-1.5*IQR
  BigMart = BigMart[(BigMart[i]>= lower_limit) & (BigMart[i]<=upper_limit)]

**Checking again after treating**

for i in BigMart.columns:
  if BigMart[i].dtype!= 'object':
    plt.boxplot(BigMart[i])
    plt.title(i)
    plt.show()

BigMart.columns

BigMart['Item_Visibility'].value_counts()

BigMart['Item_Outlet_Sales'].value_counts()

BigMart.info()

We are applying `One-hot`encoding

col_encode = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

BigMart_encoded= pd.get_dummies(BigMart,columns = col_encode, drop_first = True)

print(BigMart_encoded.head())

Encoded columns

BigMart_encoded.columns

As these columns are not needed so we are dropping them

BigMart_encoded.drop(columns = 'Item_Identifier', axis = 1, inplace = True)

BigMart_encoded.drop(columns = 'Outlet_Identifier', axis = 1, inplace = True)

BigMart_encoded.head()

BigMart_encoded.corr()

*HeatMap* visually represents the correlation between columns in dataset.
- +1 positive correlation
- -1 negative correlation
- 0 no correlation

Here we can see the `strong relationship `between **'Item_Outlet_Sales' & 'Item_MRP'**
It means high MRP tends to `high` sales of outlet.

we can see `negative co-relation` between **'Item_Outlet_Sales' & 'Item_Visibility'**
It means if things are not visible properly , sales will `decrease`.

plt.figure(figsize=(20,20))
sns.heatmap(BigMart_encoded.corr(), annot=True, cmap= 'coolwarm')

###Splitting the into dependent and independent variables

X = BigMart_encoded.drop('Item_Outlet_Sales', axis = 1)

y = BigMart_encoded['Item_Outlet_Sales']

X

y

Spliting the dataset

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state = 42)

x_train

y_train

x_test

y_test

Applying Linear Regression

from sklearn.linear_model import LinearRegression
Lr_model = LinearRegression()

Lr_model.fit(x_train,y_train)

y_pred = Lr_model.predict(x_test)
y_pred

from sklearn.metrics import *

r2_score(y_test,y_pred)

# y_train_pred = Lr_model.predict(x_train)
# r2_score(y_train,y_train_pred)

np.sqrt(mean_squared_error(y_test,y_pred))

mean_squared_error(y_test,y_pred)

Applying Random Forest algorithm for better model performance

from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestRegressor(n_estimators=50,random_state= 100)

rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)

r2_score(y_test, y_pred_rf)

np.sqrt(mean_squared_error(y_test,y_pred_rf))

Applying Hyperparameter Tuning to the model

# #Using GridSearchCV
# from sklearn.model_selection import GridSearchCV
# from scipy.stats import randint

# # Define the parameter grid with lists or numpy arrays for each hyperparameter
# param_grid = {
#     'n_estimators': randint.rvs(100, 500, size=5), # Generate 5 random values between 100 and 500
#     'max_depth': randint.rvs(5, 30, size=5), # Generate 5 random values between 5 and 30
#     'min_samples_split': randint.rvs(2, 11, size=5), # Generate 5 random values between 2 and 11
#     'min_samples_leaf': randint.rvs(1, 5, size=5), # Generate 5 random values between 1 and 5
#     'max_features': ['sqrt', 'log2', 'auto', None]
# }


# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
# grid_search.fit(x_train, y_train)
# best_params = grid_search.best_params_
# best_rf_model = grid_search.best_estimator_
# y_pred_best = best_rf_model.predict(x_test)
# r2_best = r2_score(y_test, y_pred_best)

# using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# define a parameter grid with list or numpy arrays for each hyperparameter
param_grid = {
    'n_estimators' : randint(100, 500), # Number of trees in RF
    'max_depth' : randint(5, 30), # maximum depth of the tree
    'min_samples_split' : randint(2, 11), # Minimum number of samples required to split an internal node
    'min_samples_leaf' : randint(1, 5), # minimum number of samples required to be at a leaf node
    'max_features' : ['sqrt', 'log2', 'auto', None] # Number of features to consider when looking for the best split

    }

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator = rf_model,
                                   param_distributions=param_grid,
                                   n_iter = 8,  # Number of parameter settings that are sampled
                                   cv =5, # Number of folds in cross validation,
                                   n_jobs=-1, # Use all available cores for parallel processing
                                   random_state=42) #set random state for reproducibility

# Fit the RandomizedSearchCV object to the training data
random_search.fit(x_train, y_train)

# get the best hyperparameters and the best model
best_params = random_search.best_params_
best_rf_model = random_search.best_estimator_

# Evaluate the model with best hyperparameters
y_pred_best = best_rf_model.predict(x_test)
r2_best = r2_score(y_test, y_pred_best)
print("Best Hyperparameters: \n", best_params)
print("R-squared: ", r2_best)


