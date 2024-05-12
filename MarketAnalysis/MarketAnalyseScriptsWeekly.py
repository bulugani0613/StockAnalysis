import sys
import pandas as pd
#import yfinance
from datetime import datetime
import os 
from os import path
import numpy as np
import requests
import time
import logging

start = datetime.now() 
#ScriptList contains the list of script requiring analysis 
#print('Enter your filename:')
#filename = input()
n = len(sys.argv)
filename = ''
if (n == 3):
    filename = sys.argv[1]
    NumberOfYears = int(sys.argv[2])
else:
    filename = 'FullStockList.txt'  
    #filename = 'StockConsidered.txt'
    #filename = 'StockAbove15.txt'
    #filename = 'TestStock.txt'  
    #filename = 'FilteredStock.txt'
    NumberOfYears = 10

print(filename,NumberOfYears)
print(sys.argv)

stocklistFileHandler = open(filename)
stocklist = set()

#Adding script to a set for iteration
for line in stocklistFileHandler:
    if len(line.strip()) > 0:
        stocklist.add(line.strip())

stocklistFileHandler.close() 

fileExt = datetime.today().strftime('%Y%m%d%H%M%S')
folderExt = "./data/"+datetime.today().strftime('%Y%m%d')

#Creating output folder if not exist the format is YYYYMMDDHHMMSS
try:
    os.mkdir(folderExt)
except OSError:
    print ("Creation of the directory %s failed" % folderExt)
else:
    print ("Successfully created the directory %s " % folderExt)

logging.basicConfig(filename = './' + folderExt + '/file' + str(fileExt) + '.log',
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')



#Dataframe where the summary for the stock is stored
analysisResult = pd.DataFrame(columns = ['Stock', 'Date', 'Open', 'Close', 'High', 'Low', 'PrevLow', 'DoublePrevLow', 'Volume', 'V5EMA', 'V26EMA', 'Close11EMA', 'Close21EMA', 'Close63EMA', 'Close189EMA', 'Close21EMASlope', 'Close63EMASlope', 'Close189EMASlope', 'FastMACD', 'SignalMACD', 'MACDHist', 'MACDHistSlope', 'FI', 'ADX', 'RSI', 'ImpulseColor', 'ImpulseColorPrev', 'ImpulseColorDoublePrev', 'ImpulseBuy', 'MACDHistSlopeCrossover','DaysSinceLow189', 'DaysSinceHigh189', 'RecordCount', 'low_tail', 'low_size', 'high_tail', 'high_size', 'body_height','tail_ratio','EFI', 'EFI-Inc','Last5_Mean', 'Bearish_divergence', 'Bullish_divergence', 'High63Range', 'High189Range', 'Profit63', 'Profit189', 'lowcheck', 'neg_count', 'FCloseChange', 'FCloseChangePC'])

    

#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"


def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

    
def getStockData(stock, currenttime, lasttime):
    #url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1D&from=1506996056&to="+str(currenttime)
    url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1W&from="+str(lasttime) +"&to="+str(currenttime)
    #print(url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}, verify=False)
    #print(stock, url, "*********************")
    message = stock + " " + url
    logging.info(message)
    data = response.json()
    #print('Response is: ',data)
    df = pd.DataFrame(columns = ["TimeStamp", "Date", "Open", "Close", "High", "Low", "Volume"])
    df['TimeStamp'] = data.get('t')
    df['Open'] = data.get('o')
    df['Close'] = data.get('c')
    df['High'] = data.get('h')
    df['Low'] = data.get('l')
    df['Volume'] = data.get('v')
    df['Date'] = df['TimeStamp'].apply(lambda x: getDate(x))
    #df.set_index("Date", inplace = True)
    #print(df.shape, "************************* 1111")
    #print(df)
    return df
"""
def getStockData(stock, currenttime, lasttime):
    #url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1D&from=1506996056&to="+str(currenttime)
    #url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1W&from="+str(lasttime) +"&to="+str(currenttime)
    url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1D&from="+str(lasttime) +"&to="+str(currenttime)
    #print(url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}, verify=False)
    #print(stock, url, "*********************")
    message = stock + " " + url
    logging.info(message)
    data = response.json()
    #print('Response is: ',data)
    df = pd.DataFrame(columns = ["TimeStamp", "Date", "Open", "Close", "High", "Low", "Volume"])
    df['TimeStamp'] = data.get('t')
    df['Open'] = data.get('o')
    df['Close'] = data.get('c')
    df['High'] = data.get('h')
    df['Low'] = data.get('l')
    df['Volume'] = data.get('v')
    df['Date'] = df['TimeStamp'].apply(lambda x: getDate(x))
    df.set_index('Date', inplace=True)
    #    summary = 
    weekly_summary = df.resample('W').agg(
        {
            'Open': 'first',  # First open price of the week
            'High': 'max',    # Maximum high price of the week
            'Low': 'min',     # Minimum low price of the week
            'Close': 'last',  # Last close price of the week
            'Volume': 'sum'   # Sum of volumes for the week
        }
    )
    weekly_summary.reset_index(inplace=True)
    #df.set_index('Timestamp', inplace=True)
    #weekly_summary.rename(columns={'index': 'Date'}, inplace=True)
    #df.set_index("Date", inplace = True)
    #print(df.shape, "************************* 1111")
    #print(df)
    return weekly_summary
"""

def getCurrMACDBearishDivergence(df):
    crossover_points = df[df['MACDHistCrossover'] > 0].tail(3).index
    #print(crossover_points)
    columns_to_extract = ['High', 'MACDHist']
    first_set = df.loc[crossover_points[0]:crossover_points[1] - 1, columns_to_extract]
    second_set = df.loc[crossover_points[1]:crossover_points[2] - 1, columns_to_extract]
    third_set = df.loc[crossover_points[2]:, columns_to_extract]
    bearish_divergence = 0 
    fs_positive = (first_set['MACDHist'] > 0).all()
    ss_negative = (second_set['MACDHist'] < 0).all()
    ts_positive = (third_set['MACDHist'] > 0).all()
    fs_high = first_set['High'].max()
    ts_high = third_set['High'].max()
    fs_macdhist = first_set['MACDHist'].max()
    ts_macdhist = third_set['MACDHist'].max()
    message = "MACD Bullish Divergence" +", " +  str(fs_positive) +", " + str(ss_negative) +", " + str(ts_positive) +", " + str(fs_high) +", " + str(ts_high) +", " + str(fs_macdhist) +", " + str(ts_macdhist)
    #logging.info(message)
    #print(message)
    if (fs_positive and ss_negative and ts_positive and first_set.shape[0] > 5 and second_set.shape[0] > 5 and third_set.shape[0] > 5 and (fs_high < ts_high) and (fs_macdhist > ts_macdhist)):
        bearish_divergence = 1
    else:
        bearish_divergence = 0
    return bearish_divergence

def getCurrMACDBullishDivergence(df):
    crossover_points = df[df['MACDHistCrossover'] > 0].tail(3).index
    #print(crossover_points)
    columns_to_extract = ['Low', 'MACDHist']
    first_set = df.loc[crossover_points[0]:crossover_points[1] - 1, columns_to_extract]
    second_set = df.loc[crossover_points[1]:crossover_points[2] - 1, columns_to_extract]
    third_set = df.loc[crossover_points[2]:, columns_to_extract]
    bullish_divergence = 0 
    fs_negative = (first_set['MACDHist'] < 0).all()
    ss_positive = (second_set['MACDHist'] > 0).all()
    ts_negative = (third_set['MACDHist'] < 0).all()
    fs_low = first_set['Low'].min()
    ts_low = third_set['Low'].min()
    fs_macdhist = first_set['MACDHist'].min()
    ts_macdhist = third_set['MACDHist'].min()
    message = "MACD Bullish Divergence" +", " +  str(fs_negative) +", " + str(ss_positive) +", " + str(ts_negative) +", " + str(fs_low) +", " + str(ts_low) +", " + str(fs_macdhist) +", " + str(ts_macdhist)
    #logging.info(message)
    #print(message)
    if (fs_negative and ss_positive and ts_negative and first_set.shape[0] > 5 and second_set.shape[0] > 5 and third_set.shape[0] > 5 and (fs_low > ts_low) and (fs_macdhist < ts_macdhist)):
        bullish_divergence = 1
    else:
        bullish_divergence = 0
    return bullish_divergence


    
def getDate(timestampint):
    timestamp = datetime.fromtimestamp(timestampint)
    return timestamp
    #return timestamp.strftime('%Y-%m-%d')
currenttime = int(datetime.now().timestamp())
lasttime = currenttime - (86400 * 366 * NumberOfYears)

def determine_impulse_buy(row):
    if row['ImpulseColor'] == 'Green':
        if row['ImpulseColorPrev'] in ['Blue', 'Red']:
            return 'Buy'
    elif row['ImpulseColor'] in ['Blue', 'Red']:
        return 'Sell'
    return ''

def count_consecutive_negatives(df, column_name):
    count = 0
    for value in reversed(df[column_name]):
        if value < 0:
            count += 1
        else:
            break
    return count

def determine_color(row):
    if row['MACDHistSlope'] > 0 and row['Close21EMASlope'] > 0:
        return 'Green'
    elif row['MACDHistSlope'] < 0 and row['Close21EMASlope'] < 0:
        return 'Red'
    else:
        return 'Blue'

consolidateddf = pd.DataFrame()

totalstocks = len(stocklist)
currentstock = 1
#For each stock the data is retrieved and analysis is performed. 
for stock in stocklist:
    message = 'Weekly Processing for stock ' + str(currentstock) + '/' + str(totalstocks)+ ':' + stock
    logging.info(message)
    print(message)
    currentstock = currentstock + 1
    try:
    #ticker = yfinance.Ticker(stock)
    #origdf = ticker.history(period = historyDuration)=
    #print(milliseconds)
        #currenttime = 1711411200
        origdf = getStockData(stock,currenttime, lasttime)
        #print('Received df') 
        #print(type(origdf))
        #print(origdf)
        df = origdf.copy()
        record_count = origdf.shape[0]
        row_start = record_count - 31
        row_end = record_count-1
        minmaxdf1 = origdf[row_start:row_end]
        maxId = minmaxdf1['High'].idxmax()
        minId = minmaxdf1['Low'].idxmin()
        minRec = minmaxdf1.loc[minId]
        maxRec = minmaxdf1.loc[maxId]
        
        
        #Calculating the values for EMA - 12, EMA - 26, MACD, SignalMACD, Slope of Close, EMA - 12, EMA - 26, FastMACD and SignalMACD
        df['V5EMA'] = df.Volume.ewm(span=5, adjust=False).mean()
        df['V26EMA'] = df.Volume.ewm(span=26, adjust=False).mean()
        df['Close11EMA'] = df.Close.ewm(span=11, adjust=False).mean()
        df['Close21EMA'] = df.Close.ewm(span=21, adjust=False).mean()
        df['Close21EMASlope'] = df['Close21EMA'].diff()
        df['Close63EMA'] = df.Close.ewm(span=63, adjust=False).mean()
        df['Close63EMASlope'] = df['Close63EMA'].diff()
        df['Close189EMA'] = df.Close.ewm(span=189, adjust=False).mean()
        df['Close189EMASlope'] = df['Close189EMA'].diff()
        df['Close12EMA'] = df.Close.ewm(span=12, adjust=False).mean()
        df['Close26EMA'] = df.Close.ewm(span=26, adjust=False).mean()
        df['FClose'] = df['Close'].rolling(window=4, min_periods=1).max()
        df['FClose']=df['FClose'].shift(4)
        df['FCloseChange'] = df['FClose'] - df['Close']
        df['FCloseChangePC'] = df['FCloseChange'] / df['Close'] * 100
        df = df.fillna(0)
        df['WeekChange'] = df['Close'] - df['Open']
        neg_count  = count_consecutive_negatives(df, 'WeekChange')
        
        df['FastMACD'] = df['Close12EMA'] - df['Close26EMA'] 
        df['SignalMACD'] = df['FastMACD'].ewm(span=9, adjust=False).mean()
        
        df['MACDHist'] = df['FastMACD'] - df['SignalMACD'] 
        df['MACDHistSlope'] = df['MACDHist'].diff()
        df['FI'] = (df['Close'] - df['Close'].shift(1))*df.Volume ##Force Index


        #MACD Cross overs
        positive_crossovers = df.loc[(df['FastMACD'] > df['SignalMACD']) & (df['FastMACD'].shift(1) <= df['SignalMACD'].shift(1))]
        positive_values = df.loc[(df['FastMACD'] > df['SignalMACD']) & (df['FastMACD'].shift(1) > df['SignalMACD'].shift(1))]
        negative_crossovers = df.loc[(df['FastMACD'] < df['SignalMACD']) & (df['FastMACD'].shift(1) >= df['SignalMACD'].shift(1))]
        negative_values = df.loc[(df['FastMACD'] < df['SignalMACD']) & (df['FastMACD'].shift(1) < df['SignalMACD'].shift(1))]
        df['MACDHistCrossover'] = 0
        df.loc[positive_crossovers.index, 'MACDHistCrossover'] = 1
        df.loc[negative_crossovers.index, 'MACDHistCrossover'] = 2
        
        #Calculating the ADX for the stock 
        # TR
        alpha = 1/13
        adxdf = origdf.copy()
        adxdf['H-L'] = adxdf['High'] - adxdf['Low']
        adxdf['H-C'] = np.abs(adxdf['High'] - adxdf['Close'].shift(1))
        adxdf['L-C'] = np.abs(adxdf['Low'] - adxdf['Close'].shift(1))
        adxdf['TR'] = adxdf[['H-L', 'H-C', 'L-C']].max(axis=1)
        del adxdf['H-L'], adxdf['H-C'], adxdf['L-C']

        adxdf['ATR'] = adxdf['TR'].ewm(alpha = alpha, adjust=False).mean()
        adxdf['H-pH'] = adxdf['High'] - adxdf['High'].shift(1)
        adxdf['pL-L'] = adxdf['Low'].shift(1) - adxdf['Low']
        adxdf['+DX'] = np.where(
            (adxdf['H-pH'] > adxdf['pL-L']) & (adxdf['H-pH']>0),
            adxdf['H-pH'],
            0.0
        )
        adxdf['-DX'] = np.where(
            (adxdf['H-pH'] < adxdf['pL-L']) & (adxdf['pL-L']>0),
            adxdf['pL-L'],
            0.0
        )
        del adxdf['H-pH'], adxdf['pL-L']

        # +- DMI
        adxdf['S+DM'] = adxdf['+DX'].ewm(alpha = alpha, adjust=False).mean()
        adxdf['S-DM'] = adxdf['-DX'].ewm(alpha = alpha, adjust=False).mean()
        adxdf['+DMI'] = (adxdf['S+DM']/adxdf['ATR'])*100
        adxdf['-DMI'] = (adxdf['S-DM']/adxdf['ATR'])*100
        del adxdf['S+DM'], adxdf['S-DM']

        # ADX
        adxdf['DX'] = (np.abs(adxdf['+DMI'] - adxdf['-DMI'])/(adxdf['+DMI'] + adxdf['-DMI']))*100
        adxdf['ADX'] = adxdf['DX'].ewm(alpha = alpha, adjust=False).mean()
        del adxdf['DX'], adxdf['ATR'], adxdf['TR'], adxdf['-DX'], adxdf['+DX'], adxdf['+DMI'], adxdf['-DMI']


        df['ADX'] = adxdf['ADX']
        
        # Calculating RSI
        n = 14
        rsidf = origdf.copy()
        rsidf['change']= rsidf['Close'].diff()
        rsidf['gain'] = rsidf.change.mask(rsidf.change < 0, 0.0)
        rsidf['loss'] = -rsidf.change.mask(rsidf.change > 0, -0.0)
        rsidf['avg_gain'] = rma(rsidf.gain[n+1:].to_numpy(), n, np.nansum(rsidf.gain.to_numpy()[:n+1])/n)
        rsidf['avg_loss'] = rma(rsidf.loss[n+1:].to_numpy(), n, np.nansum(rsidf.loss.to_numpy()[:n+1])/n)
        rsidf['rs'] = rsidf['avg_gain'].divide(rsidf.avg_loss)
        rsidf['Rsi_14'] = 100 - (100 / (1 + rsidf.rs))
        df['RSI'] = rsidf['Rsi_14']
        
        df['EFI'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df['EFI'] = df['EFI'].fillna(0)  # Handle NaN values for the first row  
        #df['EFI-13'] = df['EFI'].rolling(window=13, min_periods=1).sum()
        df['EFI-13'] = df['EFI'].ewm(span=13, adjust=False).mean()
        df['EFI-13Diff'] = (df['EFI-13'] - df['EFI-13'].shift(1))/df['EFI-13'].shift(1)
        
        last_low = df['Low'].tail(5)
        first_low = last_low.iloc[0]
        second_low = last_low.iloc[1]
        third_low = last_low.iloc[2]
        #last_diff = abs(first_low - third_low)/ min(first_low, third_low)
        #Find the max of low with current low for 3 weeks. 
        #Find the min of low with current low for 3 weeks. 
        #Find the difference - this could be a good stock 
        last_low = last_low.iloc[:4]
        max_last_low = max(last_low)
        min_last_low = min(last_low)
        last_diff = (max_last_low - min_last_low) / min_last_low
        
        df['Growth'] = (df['High'] - df['Low'])/ df['Low']*100
        last_5 = df['Growth'][-5:]
        condition_met = (last_5 > 5).all()
        
        average_5 = 0
        if condition_met:
            average_5 = last_5.mean()
        
        
        df['ImpulseColor'] = df.apply(determine_color, axis=1)
        df['ImpulseColorPrev'] = df['ImpulseColor'].shift()
        df['ImpulseBuy'] = df.apply(determine_impulse_buy, axis=1)
        
        positive_crossovers = None
        negative_crossovers = None
        
        positive_crossovers = df.loc[(df['MACDHistSlope'] > 0) & (df['MACDHistSlope'].shift(1) <= 0) & (df['MACDHist'] < 0)]
        negative_crossovers = df.loc[(df['MACDHistSlope'] < 0 ) & (df['MACDHistSlope'].shift(1) >= 0) & (df['MACDHist'] > 0)]
        df['MACDHistSlopeCrossover'] = 0
        df.loc[positive_crossovers.index, 'MACDHistSlopeCrossover'] = 1
        df.loc[negative_crossovers.index, 'MACDHistSlopeCrossover'] = 2
        
        lastRow = df.shape[0]
        ImpulseColorDoublePrev = df.iloc[lastRow - 3]['ImpulseColor']
        currentStockValues = df.iloc[lastRow-1:]
        previousLow=df.iloc[lastRow-2]['Low']
        doublepreviousLow=df.iloc[lastRow-3]['Low']
        
        current_open = currentStockValues['Open'].values[0]
        current_close = currentStockValues['Close'].values[0]
        current_high = currentStockValues['High'].values[0]
        current_low = currentStockValues['Low'].values[0]
        
        #print(current_low, current_close, current_high, current_low)
        
        current_min = min(current_open, current_close)
        current_max = min(current_open, current_close)
        
        #print(current_min, current_max)
        
        low_tail = (current_min - current_low) / current_low * 100
        high_tail = (current_high - current_max ) / current_max * 100
        low_size = (current_min - current_low ) / (current_max - current_low) * 100
        high_size = (current_high - current_max ) / (current_max - current_low) * 100
        
        tail_ratio = low_tail / high_tail
        
        body_height = (current_close - current_open)/(current_high - current_low)
        
        High63DaysMax = df['High'].tail(13).max()
        High189DaysMax = df['High'].tail(52).max() 
        High63DaysMin = df['Low'].tail(13).min()
        High189DaysMin = df['Low'].tail(52).min() 
        
        High63Range = (currentStockValues['Close'].values[0] - High63DaysMin)/(High63DaysMax - High63DaysMin)
        High189Range = (currentStockValues['Close'].values[0] - High189DaysMin)/(High189DaysMax - High189DaysMin)
        Profit63 = (High63DaysMax - currentStockValues['Close'].values[0])/currentStockValues['Close'].values[0]
        Profit189 = (High189DaysMax - currentStockValues['Close'].values[0])/currentStockValues['Close'].values[0]
        #print(High63DaysMax)
        #print(Profit63)
        #print(High189DaysMax)
        #print(Profit189)
        
        
        high189_index = df['High'].idxmax()
        low189_index = df['Low'].idxmin()

        # Retrieve the records with the minimum and maximum 'Age'
        record_low189 = df.loc[low189_index]
        record_high189 = df.loc[high189_index]
        
        DaysSinceLow189 = (datetime.now() - record_low189['Date']).days
        DaysSinceHigh189 = (datetime.now() - record_high189['Date']).days
        
        #df['High4Week'] = 
        df['High4Week'] = df['Close'].rolling(window=4, min_periods=1).max()
        df['Buy'] = np.where(df['High4Week'] > df['Close'] * 1.15, 1, 0)
        Bearish_divergence = getCurrMACDBearishDivergence(df)
        Bullish_divergence = getCurrMACDBullishDivergence(df)
        
        new_record = {
            'Date':currentStockValues['Date'].values[0],
            'Open':currentStockValues['Open'].values[0],
            'Close':currentStockValues['Close'].values[0],
            'High':currentStockValues['High'].values[0],
            'Low':currentStockValues['Low'].values[0],
            'PrevLow' : previousLow, 
            'DoublePrevLow' : doublepreviousLow,
            'Volume':currentStockValues['Volume'].values[0],
            'V5EMA':currentStockValues['V5EMA'].values[0],
            'V26EMA':currentStockValues['V26EMA'].values[0],
            'Close11EMA':currentStockValues['Close11EMA'].values[0],
            'Close21EMA':currentStockValues['Close21EMA'].values[0],
            'Close63EMA':currentStockValues['Close63EMA'].values[0],
            'Close189EMA':currentStockValues['Close189EMA'].values[0],
            'Close21EMASlope':currentStockValues['Close21EMASlope'].values[0],
            'Close63EMASlope':currentStockValues['Close21EMASlope'].values[0],
            'Close189EMASlope':currentStockValues['Close21EMASlope'].values[0],
            'FastMACD':currentStockValues['FastMACD'].values[0],
            'SignalMACD':currentStockValues['SignalMACD'].values[0],
            'MACDHist':currentStockValues['MACDHist'].values[0],
            'MACDHistSlope':currentStockValues['MACDHistSlope'].values[0],
            'FI':currentStockValues['FI'].values[0],
            'ADX':currentStockValues['ADX'].values[0],
            'RSI':currentStockValues['RSI'].values[0],
            'ImpulseColor':currentStockValues['ImpulseColor'].values[0],
            'ImpulseColorPrev':currentStockValues['ImpulseColorPrev'].values[0],
            'ImpulseBuy':currentStockValues['ImpulseBuy'].values[0],
            'ImpulseColorDoublePrev': ImpulseColorDoublePrev,
            'MACDHistSlopeCrossover':currentStockValues['MACDHistSlopeCrossover'].values[0],
            'Stock' : stock,
            'High63Range' : High63Range,
            'High189Range' : High189Range,
            'Profit63' : Profit63,
            'Profit189' : Profit189,
            'DaysSinceLow189' : DaysSinceLow189, 
            'DaysSinceHigh189' : DaysSinceHigh189,
            'RecordCount' : record_count,
            'low_tail' : low_tail, 
            'low_size' : low_size,
            'high_tail' : high_tail,
            'high_size' : high_size,
            'body_height' :body_height,
            'tail_ratio' : tail_ratio,
            'EFI':currentStockValues['EFI-13'].values[0],
            'Last5_Mean': average_5,
            'EFI-Inc':currentStockValues['EFI-13Diff'].values[0],
            'Bearish_divergence' : Bearish_divergence,
            'Bullish_divergence' : Bullish_divergence,
            'lowcheck' : last_diff,
            'neg_count' : neg_count,
            'FCloseChange' : currentStockValues['FCloseChange'].values[0],
            'FCloseChangePC' : currentStockValues['FCloseChangePC'].values[0]}
        
        
        
        
        #analysisResult = analysisResult.append( , ignore_index = True)
        #positiveindex, negativeindex, currentindex
        first_line = df.iloc[0]
        if currentstock == 1:
            logging.info(first_line)
        
        df['StockName'] = stock
        #list_df = [consolidateddf, df]
        #consolidateddf= pd.concat(list_df)
        new_record_df = pd.DataFrame([new_record])
        #print(new_record_df)
        analysisResult = pd.concat([analysisResult, new_record_df], ignore_index=True)
        #print(analysisResult)
        #df.to_csv("./AnalysisResult/" +stock+str(fileExt) + ".csv")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        message = str(sys.exc_info()[0]) + "occurred while processing for stock " + stock + ". Proceeding to next stock. Error is in " + str(exc_type) + fname + str(exc_tb.tb_lineno)
        logging.exception(message)  
        print(message)
        

    #df.to_csv("./" + folderExt + "/" +stock+str(fileExt) + ".csv")
    #consolidateddf= consolidateddf.append(df)
    #consolidateddf= pd.concat([consolidateddf, df], ignore_index=True)
    #print(consolidateddf.size)

#analysisResult['Consider'] = df['12EMA'] >df['26EMA'] and df['26EMA'] > df['63EMA'] and df['63EMA'] > df['200EMA'] and df['12EMASlope']>0 and df['26EMASlope']>0 and df['63EMASlope']>0 and df['200EMASlope']>0 and df['FastMACD']>df['SignalMACD']     
#Rule - Tail ratio greater than 3; High 63 is less than 0.3 
filtered_df = analysisResult[((analysisResult['MACDHistSlopeCrossover'] == 1) & (analysisResult['Volume'] > 500000) & (analysisResult['SignalMACD'] < 0) & (analysisResult['Close'] > 50)) | ((analysisResult['ImpulseBuy'] == 'Buy') & (analysisResult['Volume'] > 500000)) | ((analysisResult['tail_ratio'] > 2) & (analysisResult['Volume'] > 500000))]
tail_df = analysisResult[((analysisResult['tail_ratio'] > 3) & (analysisResult['Volume'] > 500000) & (analysisResult['High63Range'] < 0.3))]
tail_df.to_csv('./AnalysisResult/WeeklyTailAnalysisResult' + str(fileExt) + '.csv')
column_to_save = filtered_df['Stock']
column_to_save.to_csv('FilteredStock.txt', index=False, header=False)
filtered_df.to_csv('./AnalysisResult/WeeklyAnalysisResult' + str(fileExt) + '.csv')
analysisResult.to_csv('./AnalysisResult/WeeklyFullAnalysisResult' + str(fileExt) + '.csv')
#consolidateddf.to_csv('./AnalysisResult/ConsolidatedAnalysisResult' + str(fileExt) + '.csv')
end = datetime.now() 
message = 'Time taken (sec):' + str((end-start).seconds)
print(message)
logging.info(message)