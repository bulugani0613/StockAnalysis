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
    #filename = 'FullStockList.txt'  
    #filename = 'StockAbove15.txt'
    filename = 'TestStock.txt'  
    #filename = 'FilteredStock.txt'
    NumberOfYears = 25

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
analysisResult = pd.DataFrame(columns = ['Stock', 'TimeStamp', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'V5EMA', 'V26EMA', 'Close11EMA', 'Close21EMA', 'Close21EMASlope', 'Close63EMA', 'Close189EMA', 'FastMACD', 'SignalMACD', 'MACDHist', 'MACDHistSlope', 'FI', 'ADX', 'RSI', 'ImpulseColor', 'ImpulseColorPrev', 'ImpulseBuy', 'MACDHistSlopeCrossover', 'High63Range', 'High189Range', 'Bearish_divergence', 'Bullish_divergence'])

    

#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"


def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

def getCurrMACDBearishDivergence(df):
    crossover_points = df[df['MACDHistCrossover'] > 0].tail(3).index
    #print(crossover_points)
    columns_to_extract = ['High', 'MACDHist']
    #print(df)
    #print(crossover_points)
    #print(df.loc[crossover_points[0]:crossover_points[1] - pd.Timedelta(days=1)])
    first_set = df.loc[crossover_points[0]:crossover_points[1] - pd.Timedelta(days=1), columns_to_extract]
    second_set = df.loc[crossover_points[1]:crossover_points[2] - pd.Timedelta(days=1), columns_to_extract]
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
    first_set = df.loc[crossover_points[0]:crossover_points[1] - pd.Timedelta(days=1), columns_to_extract]
    second_set = df.loc[crossover_points[1]:crossover_points[2] - pd.Timedelta(days=1), columns_to_extract]
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
    #df.set_index("Date", inplace = True)
    #print(df.shape, "************************* 1111")
    #print(df)
    return df
    
def getDate(timestampint):
    timestamp = datetime.fromtimestamp(timestampint)
    return timestamp
    #return timestamp.strftime('%Y-%m-%d')
currenttime = int(datetime.now().timestamp())
lasttime = currenttime - (86400 * 366 * NumberOfYears)

def determine_impulse_buy(row):
    if row['ImpulseColor'] == 'Green':
        if row['ImpulseColorPrev'] in ['Red']:
            return 'Buy'
    elif row['ImpulseColor'] in ['Blue', 'Red']:
        return 'Sell'
    return ''

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
        origdf = getStockData(stock,currenttime, lasttime)
        #print('Received df') 
        #print(type(origdf))
        #print(origdf)
        df = origdf.copy()
 
        
        
        #Calculating the values for EMA - 12, EMA - 26, MACD, SignalMACD, Slope of Close, EMA - 12, EMA - 26, FastMACD and SignalMACD
        df['V5EMA'] = df.Volume.ewm(span=5, adjust=False).mean()
        df['V26EMA'] = df.Volume.ewm(span=26, adjust=False).mean()
        df['Close11EMA'] = df.Close.ewm(span=11, adjust=False).mean()
        df['Close21EMA'] = df.Close.ewm(span=21, adjust=False).mean()
        df['Close21EMASlope'] = df['Close21EMA'].diff()
        df['Close63EMA'] = df.Close.ewm(span=63, adjust=False).mean()
        df['Close189EMA'] = df.Close.ewm(span=189, adjust=False).mean()
        df['FastMACD'] = df['Close11EMA'] - df['Close21EMA'] 
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
        currentStockValues = df.iloc[lastRow-1:]
        
        High63DaysMax = df['High'].tail(63).max()
        High189DaysMax = df['High'].tail(189).max() 
        High63DaysMin = df['High'].tail(63).min()
        High189DaysMin = df['High'].tail(189).min() 
        
        High63Range = (currentStockValues['High'].values[0] - High63DaysMin)/(High63DaysMax - High63DaysMin)
        High189Range = (currentStockValues['High'].values[0] - High189DaysMin)/(High189DaysMax - High189DaysMin)
        
        Bearish_divergence = getCurrMACDBearishDivergence(df)
        Bullish_divergence = getCurrMACDBullishDivergence(df)
        
        
        new_record = {'TimeStamp':currentStockValues['TimeStamp'].values[0],
            'Open':currentStockValues['Open'].values[0],
            'Close':currentStockValues['Close'].values[0],
            'High':currentStockValues['High'].values[0],
            'Low':currentStockValues['Low'].values[0],
            'Volume':currentStockValues['Volume'].values[0],
            'V5EMA':currentStockValues['V5EMA'].values[0],
            'V26EMA':currentStockValues['V26EMA'].values[0],
            'Close11EMA':currentStockValues['Close11EMA'].values[0],
            'Close21EMA':currentStockValues['Close21EMA'].values[0],
            'Close21EMASlope':currentStockValues['Close21EMASlope'].values[0],
            'Close63EMA':currentStockValues['Close63EMA'].values[0],
            'Close189EMA':currentStockValues['Close189EMA'].values[0],
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
            'MACDHistSlopeCrossover':currentStockValues['MACDHistSlopeCrossover'].values[0],
            'Stock' : stock,
            'High63Range' : High63Range,
            'High189Range' : High189Range,
            'Bearish_divergence' : Bearish_divergence,
            'Bullish_divergence' : Bullish_divergence}
        
        
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
#filtered_df = analysisResult[((analysisResult['MACDHistSlopeCrossover'] == 1) & (analysisResult['Volume'] > 100000) & (analysisResult['SignalMACD'] < 0) & (analysisResult['Close'] > 50)) | (analysisResult['ImpulseBuy'] == 'Buy')]
#column_to_save = filtered_df['Stock']
#column_to_save.to_csv('FilteredStock.txt', index=False, header=False)
analysisResult.to_csv('./AnalysisResult/DailyAnalysisResult' + str(fileExt) + '.csv')
#consolidateddf.to_csv('./' + folderExt  + '/' + filename + "ConsolidatedResult" + str(fileExt) + ".csv")
end = datetime.now() 
message = 'Time taken (sec):' + str((end-start).seconds)
print(message)
logging.info(message)