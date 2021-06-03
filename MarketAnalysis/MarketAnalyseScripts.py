import sys
import pandas as pd
import yfinance
from datetime import datetime
import os 
from os import path
import numpy as np


#ScriptList contains the list of script requiring analysis 
stocklistFileHandler = open("ScriptList.txt")
stocklist = set()

#Adding script to a set for iteration
for line in stocklistFileHandler:
    if len(line.strip()) > 0:
        stocklist.add(line.strip())

stocklistFileHandler.close() 

fileExt = datetime.today().strftime('%Y%m%d%H%M%S')
folderExt = datetime.today().strftime('%Y%m%d%H%M%S')

#Creating output folder if not exist the format is YYYYMMDDHHMMSS
try:
    os.mkdir(folderExt)
except OSError:
    print ("Creation of the directory %s failed" % folderExt)
else:
    print ("Successfully created the directory %s " % folderExt)

#Dataframe where the summary for the stock is stored
analysisResult = pd.DataFrame(columns = ['Stock', 'Close', '12EMA', '26EMA', 'FastMACD', 'SignalMACD', 'MACDHist', 'ADX', 'RSI', 'CloseSlope', '12EMASlope', '26EMASlope', 'FastMACDSlope', 'SignalMACDSlope', 'MACDHistSlope', 'ADXSlope', 'PositiveIndicator'])


#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"

def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

#For each stock the data is retrieved and analysis is performed. 
for stock in stocklist:
    print('Processing ', stock)
    try:
        ticker = yfinance.Ticker(stock)
        origdf = ticker.history(period = historyDuration)
        df = origdf.copy()
        
        #Calculating the values for EMA - 12, EMA - 26, MACD, SignalMACD, Slope of Close, EMA - 12, EMA - 26, FastMACD and SignalMACD
        df['12EMA'] = df.Close.ewm(span=12, adjust=False).mean()
        df['26EMA'] = df.Close.ewm(span=26, adjust=False).mean()
        df['FastMACD'] = df['12EMA'] - df['26EMA'] 
        df['SignalMACD'] = df['FastMACD'].ewm(span=9, adjust=False).mean()
        df['MACDHist'] = df['FastMACD'] - df['SignalMACD'] 

        df['CloseSlope'] = df.Close.diff()
        df['12EMASlope'] = df['12EMA'].diff()
        df['26EMASlope'] = df['26EMA'].diff()
        df['FastMACDSlope'] = df['FastMACD'].diff()
        df['SignalMACDSlope'] = df['SignalMACD'].diff()
        df['MACDHistSlope'] = df['SignalMACD'].diff()
        
        

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
        df['ADXSlope'] = df['ADX'].diff()
        
        
        # Calculating RSI
        n = 14
        rsidf = origdf.copy()
        rsidf['change']= rsidf['Close'].diff()
        rsidf['gain'] = rsidf.change.mask(rsidf.change < 0, 0.0)
        rsidf['loss'] = -rsidf.change.mask(rsidf.change > 0, -0.0)
        rsidf['avg_gain'] = rma(rsidf.gain[n+1:].to_numpy(), n, np.nansum(rsidf.gain.to_numpy()[:n+1])/n)
        rsidf['avg_loss'] = rma(rsidf.loss[n+1:].to_numpy(), n, np.nansum(rsidf.loss.to_numpy()[:n+1])/n)
        rsidf['rs'] = rsidf.avg_gain / rsidf.avg_loss
        rsidf['Rsi_14'] = 100 - (100 / (1 + rsidf.rs))
        
        df['RSI'] = rsidf['Rsi_14']


        df['PositiveIndicator'] = 0
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['CloseSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['12EMASlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['26EMASlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['FastMACDSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['SignalMACDSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['MACDHistSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['ADXSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['RSI'].apply(lambda x: 1 if x < 15 else 0)

        lastRow = df.shape[0]
        currentStockValues = df.iloc[lastRow-1:]
        
        #analysisResult = pd.DataFrame(columns = ['Stock', 'Close', '12EMA', '26EMA', 'FastMACD', 'SignalMACD', 'MACDHist', 'ADX', 'RSI', 'CloseSlope', '12EMASlope', '26EMASlope', 'FastMACDSlope', 'SignalMACDSlope', 'ADXSlope', 'PositiveIndicator'])

        analysisResult = analysisResult.append( {'Stock' : stock, 'Close' : currentStockValues['Close'][0], '12EMA': currentStockValues['12EMA'][0], '26EMA': currentStockValues['26EMA'][0], 'FastMACD': currentStockValues['FastMACD'][0], 'SignalMACD': currentStockValues['SignalMACD'][0], 'MACDHist' : currentStockValues['MACDHist'][0], 'ADX': currentStockValues['ADX'][0], 'RSI': currentStockValues['RSI'][0], 'CloseSlope': currentStockValues['CloseSlope'][0], '12EMASlope': currentStockValues['12EMASlope'][0], '26EMASlope': currentStockValues['26EMASlope'][0],  'FastMACDSlope': currentStockValues['FastMACDSlope'][0], 'SignalMACDSlope': currentStockValues['SignalMACDSlope'][0], 'MACDHistSlope' : currentStockValues['MACDHistSlope'][0],'ADXSlope': currentStockValues['ADXSlope'][0], 'PositiveIndicator' : currentStockValues['PositiveIndicator'][0]}, ignore_index = True)
    except:
        print("Oops!", sys.exc_info()[0], "occurred while processing for stock ", stock, ". Proceeding to next stock")  

    df.to_csv("./" + folderExt + "/" +stock+str(fileExt) + ".csv")
analysisResult.to_csv("AnalysisResult" + str(fileExt) + ".csv")

