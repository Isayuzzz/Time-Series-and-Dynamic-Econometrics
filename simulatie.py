#Python file

###########################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import numpy.random as rnd
import scipy.integrate as ing
import time
from datetime import timedelta
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.graphics.tsaplots as sgt
###########################################################

def readfile():
    '''
    Purpose: read a file
    input: input file 
    output: dataframe of the data 
    '''
    df = pd.read_csv('C:/Users/jipde/OneDrive/VU/Year 3/ITSDE p1/data_assign_p1 (2).csv', delimiter=',',parse_dates=True)
    
    return df

def LLR_test(model_1,model_2,DF=1):
    L1 = model_1.fit().llf
    L2 = model_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR,DF).round(3)
    
    return p 

def ex1(df):
    
    f, ax = plt.subplots(figsize=(15,5))
    ax.plot(df['obs'], df['GDP_QGR'])
    ax.set_title('Dutch GDP quarterly growth rates 1987Q2-2009Q1')
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP')
    plt.grid(True)
    plt.show()
    
    # calculate ACF and PACF upto 12 lags
    # acf_12 = acf(df['GDP_QGR'], nlags=12)
    # pacf_12 = pacf(df['GDP_QGR'], nlags=12)
    
    f, ax = plt.subplots(1,2,figsize=(10,3), dpi= 100)
    plot_acf(df['GDP_QGR'].tolist(), lags=12, ax=ax[0])
    plot_pacf(df['GDP_QGR'].tolist(), lags=12, ax=ax[1])
    
def ex2(df):
    
    # test for stationarity by ADF test
    resultsADF = adfuller(df['GDP_QGR'].dropna(), autolag='AIC',regression='n')
    print('p-value: ', resultsADF[1])
    # transforming data st it fits stationarity assumptions
    resultsADF = adfuller(df['GDP_QGR'].diff().dropna(), autolag='AIC',regression='n') 
    print('p-value: ', resultsADF[1])
    dfResults = pd.Series(resultsADF[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])

    #Add Critical Values
    for key,value in resultsADF[4].items():
        dfResults['Critical Value (%s)'%key] = value
    print('Augmented Dickey-Fuller Test Results:')
    print(dfResults)
    
    # non-stationarity
    for p in range(5):
        model_ar = ARIMA(df['GDP_QGR'],order=(p,0,0))
        results_ar = model_ar.fit()
        #print(results_ar.summary())
    model_ar_1 = ARIMA(df['GDP_QGR'],order=(1,0,0)) # 'best model', lowest AIC
    results_ar_1 = model_ar_1.fit()
    print(results_ar_1.summary())
    
    # stationarity
    for p in range(5):
        model_ar_S = ARIMA(df['GDP_QGR'],order=(p,1,0))
        results_ar_S = model_ar.fit()
        #print(results_ar.summary())
    model_ar_3S = ARIMA(df['GDP_QGR'],order=(3,0,0))  # best model, lowest AIC  
    results_ar_3S = model_ar_3S.fit()
    print(results_ar_3S.summary())
    
    return results_ar_3S
    
    '''
    train = df.iloc[:-8]
    test = df.iloc[-8:]
    print(train.shape,test.shape)
    
    model = ARIMA(train['GDP_QGR'], order=(4,1,1))
    model = model.fit()
    print(model.summary())
    # make predictions on test set
    pred = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print(pred)
    '''
    '''
    train = df['GDP_QGR'][:len(df)-8]
    test = df['GDP_QGR'][len(df)-8:]
    
    ar_model = AutoReg(train, lags=22).fit()
    print(ar_model.summary())
    
    pred = ar_model.predict(start=len(train), end=(len(df)-1), dynamic=False)
    
    f, ax = plt.subplots(figsize=(10,5))
    ax.plot(pred)
    ax.plot(test, color='red')
    
    rmse = np.sqrt(mean_squared_error(test,pred))
    print(rmse)
    
    pred_future=ar_model.predict(start=len(df)+1, end = len(df)+8,dynamic=False)
    
    print(pred_future)
    '''
   
def ex3(df,results_ar_3S):
    
    df['RES_GDP_QGR'] = results_ar_3S.resid
    
    #mean = df['RES_GDP_QGR'].mean()
    #var = df['RES_GDP_QGR'].var()
    resultsADF = adfuller(df['RES_GDP_QGR'].dropna(), autolag='AIC',regression='n')
    print('p-value: ', resultsADF[1]) # residuals are stationary 
    # plot the estimated residual ACF function
    plot_acf(df['RES_GDP_QGR'].tolist(), lags=12)
    plt.show()
    
def ex4(df,results_ar_3S):
    
    train = df.iloc[:-8]
    test = df.iloc[-8:]
    
    pred = results_ar_3S.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    print(pred)   
    
    return pred
    
def ex5(df,pred,a):
    
    print(sts.t.interval(alpha = a, df=len(pred), loc=np.mean(pred), scale=sts.sem(pred)))
    
def main():
    # Magic numbers 
    a = 0.95
    # code 
    df = readfile()
    ex1(df)
    results_ar_3S = ex2(df)
    ex3(df,results_ar_3S)
    pred = ex4(df,results_ar_3S)
    ex5(df,pred,a)
###########################################################
### call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("Execution took:",timedelta(seconds=end_time - start_time))
