import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K 
from keras.models import load_model, save_model
import pandas as pd
import pickle

def create_dataset(x,y,time_step = 1):
  xs = []
  ys = []
  for i in range(len(x) - time_step):
    v = x.iloc[i:(i+time_step)].values
    xs.append(v)
    ys.append(y.iloc[i+time_step])
  return np.array(xs), np.array(ys)


end_date = datetime.date.today()
tickers = pd.read_csv('tickers.csv')
start_date = end_date + datetime.timedelta(days=-700)
start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
# Set the ticker

ticker = int(input('Welcome to the market insights analysis. Please, choose one company to see a forecasting for the stock price (1: Google, 2: Tesla, 3: Amazon, 4: Other: '))

stocks = {1:'GOOG',2:'TSLA',3:'AMZN'}

if ticker in list(stocks.keys()):
    ticker_lab = stocks[ticker]
# Get the data
    data = yf.download(ticker_lab, start_date, end_date)
    last_price = data.iloc[-1,:]['Open']
    last_date  = datetime.datetime.strftime(data.iloc[-1].name, "%Y-%m-%d")
    cur_date = datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)
    cur_date = datetime.datetime.strftime(cur_date, "%Y-%m-%d")
    if ticker == 1:
        model = load_model("model_google.h5")

        with open('esc_google.pkl' , 'rb') as f:
            esc = pickle.load(f)
    elif ticker == 2:
        model = load_model("model_tesla.h5")

        with open('esc_tesla.pkl' , 'rb') as f:
            esc = pickle.load(f)
    elif ticker == 3:
        model = load_model("model_amazon.h5")

        with open('esc_amazon.pkl' , 'rb') as f:
            esc = pickle.load(f)
    y_value = esc.transform(last_price.reshape(1,-1))
    y_value = y_value.reshape(1,1,1)
    y_predict = model.predict(y_value)
    y_pred_conv = round(esc.inverse_transform(y_predict[0][0][0].reshape(1,-1))[0][0],2)
    print(f'According with our analysis, the stock price of {ticker_lab} for the date {cur_date}  will be $ {y_pred_conv :.2f}')

else:
    pattern = input('Insert the name of the company: ')
   
    string_list = list(tickers['Name'].values)

    regex_pattern = re.compile(rf'{re.escape(pattern)}', re.IGNORECASE)
    options = []

    for string in string_list:
        if regex_pattern.search(string):
            options.append(string)
    

    df_options = tickers[tickers['Name'].isin(options)][['Symbol','Name']].reset_index()
    while df_options.shape[0]==0:
        
        pattern = input('Sorry, I did not find options. Please try with another words: ')
        string_list = list(tickers['Name'].values)

        regex_pattern = re.compile(rf'{re.escape(pattern)}', re.IGNORECASE)
        options = []

        for string in string_list:
            if regex_pattern.search(string):
                options.append(string)
        df_options = tickers[tickers['Name'].isin(options)][['Symbol','Name']].reset_index()
    
    print(df_options)

    print('I found the following options.')

    ticker_lab = input('Please enter the symbol of the company selected: ')
    while ticker_lab not in df_options['Symbol'].values:        
        ticker_lab = input('Remember to enter the symbol. You can see the options below. If you want to exit, write exit: ')
        print(df_options)
        if ticker_lab == 'exit':
            break
    try:
        data = yf.download(ticker_lab, start_date, end_date)
        series = data[['Close']]
        test_size = 30
        index = len(series) - test_size
        train = series[:index]
        test = series[index:]
        esc = StandardScaler()
        train_esc = pd.DataFrame(esc.fit_transform(train), columns= ['Close'])
        test_esc = pd.DataFrame(esc.transform(test), columns = ['Close'])
        time_lag = 1
        x_train, y_train = create_dataset(train_esc, train_esc.Close, time_lag)
        x_test, y_test = create_dataset(test_esc, test_esc.Close, time_lag)
        K.clear_session()

        model = Sequential()
        model.add(LSTM(units = 250, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(rate =  0.05))
        model.add(LSTM(units = 250, return_sequences = True))
        model.add(Dropout(rate = 0.05))
        model.add(LSTM(units = 250, return_sequences = True))
        model.add(LSTM(units = 100, return_sequences = True))
        model.add(Dropout(rate = 0.01))
        model.add(Dense(units = 1))
        model.compile(optimizer =  'adam', loss = 'mean_squared_error')
        history = model.fit(x_train, y_train, epochs = 50, shuffle = False, batch_size = 32)
        last_price = data.iloc[-1,:]['Open']
        last_date  = datetime.datetime.strftime(data.iloc[-1].name, "%Y-%m-%d")
        cur_date = datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)
        cur_date = datetime.datetime.strftime(cur_date, "%Y-%m-%d")
        y_value = esc.transform(last_price.reshape(1,-1))
        y_predict = model.predict(y_value)
        y_pred_conv = round(esc.inverse_transform(y_predict[0][0][0].reshape(1,-1))[0][0],2)
        print(f'According with our analysis, the stock price of {ticker_lab} for the date {cur_date} will be $ {y_pred_conv :.2f}')
    except:
        print('Thank you for using the chatbot service')