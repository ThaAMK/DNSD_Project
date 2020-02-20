# DNSD_Project
Overview:

  One of the main challenges that face all business all around the world is to predict their yearly demand, that will help the business to plan their stocks, workforce, and to budget based on that. As machine learning is developing, now we can utilize it in predicting sales of the stores, based on the data we have we can go as predicting by store per department or by item. 
In this project, we analyze the sales data for Walmart all over the states and try to predict the sales per week per store per department for a sample of future dates. The project was inspired by a competition on Kaggle by Walmart. You can find it here. 

Conclusion:

Future prediction models can be very tricky because the future can be changed dramatically and totally new factors can be introduced, for example, the data here does not take in factor the online shopping effect on store sales. In problems like this, we always try to keep very sure to not overfit the data and to review the models with updated data to get a better picture.

Libraries used:

    import numpy as np # linear algebra
    import pandas as pd # data processing
    from sklearn.preprocessing import StandardScaler #for scaling
    import matplotlib.pyplot as plt # ploting
    from sklearn.metrics import mean_absolute_error #calculate error 
    from sklearn.model_selection import train_test_split #splitting the data
    from sklearn.model_selection import KFold # splitting the data
    from sklearn.neural_network import MLPRegressor
    #from google.colab import drive
    from sklearn.impute import SimpleImputer #dealing with nulls
    import seaborn as sns #plotting the data
    import datetime
    from sklearn.metrics import make_scorer #scoring
    from sklearn.model_selection import GridSearchCV #find best Model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import LinearSVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.kernel_approximation import Nystroem

Files:
stores.csv

This file contains anonymized information about the 45 stores, indicating the type and size of store.

train.csv

This is the historical training data, which covers to 2010-02-05 to 2012-11-01. Within this file you will find the following fields:

    Store - the store number
    Dept - the department number
    Date - the week
    Weekly_Sales -  sales for the given department in the given store
    IsHoliday - whether the week is a special holiday week

test.csv

This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

features.csv

This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

    Store - the store number
    Date - the week
    Temperature - average temperature in the region
    Fuel_Price - cost of fuel in the region
    MarkDown1-5 - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
    CPI - the consumer price index
    Unemployment - the unemployment rate
    IsHoliday - whether the week is a special holiday week

