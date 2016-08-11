from sklearn.linear_model import Ridge
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

DEGREE = 5
Z_SCORE = 3
poly = PolynomialFeatures(DEGREE)
INPUT_FILE = "all_data.csv"

def preprocessing():
    #remove outliers
    dataframe = pd.read_csv(INPUT_FILE)
    dataframe.sort_values(by='HR',inplace=True)
    for hour in range(0,24):
        data = dataframe.loc[dataframe.HR==hour]
        d = np.abs(data['TRAFFIC_COUNT'] - np.median(data['TRAFFIC_COUNT']))
        median_distance = np.median(d)
        absolute_distance = d/median_distance if median_distance else 0
        dataframe.loc[dataframe.HR==hour] = data[absolute_distance<Z_SCORE]
    dataframe = dataframe.dropna(how='any')
    return dataframe

def regression(dataframe):
    #shuffle data
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    #select features
    columns = ['HR','WEEK_DAY','DAY_OF_YEAR']

    features = np.array(dataframe[columns])
    target = np.array(dataframe['TRAFFIC_COUNT'])
    
    #add polynomial features, reshape input into 2D array for fitting
    poly_features = poly.fit_transform(features[0].reshape(1,-1))
    for index in range(1,int(features.size/len(columns))): #features.size returns the total number of elements, we want # of rows
    	poly_features = np.vstack([poly_features,poly.fit_transform(features[index].reshape(1,-1))])

    #split training and test sets
    train_set_X, test_set_X , train_set_Y, test_set_Y = cross_validation.train_test_split(poly_features,target,test_size = 0.3,random_state=0)
    regr = Ridge(alpha = 1, fit_intercept = True)
    regr.fit(train_set_X,train_set_Y)
    return regr



