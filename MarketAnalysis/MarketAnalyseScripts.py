import sys
import pandas as pd
import yfinance
from datetime import datetime
import os 
from os import path


#ScriptList contains the list of script requiring analysis 
stocklistFileHandler = open("ScriptList.txt")
stocklist = set()

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
analysisResult = pd.DataFrame(columns = ['Stock', 'Close', '12EMA', '26EMA', 'MACDHist', 'SignalMACD', 'CloseSlope', 'ShortEMASlope', 'LongEMASlope', 'MACDHistSlope', 'SignalMACDSlope', 'PositiveIndicator'])


#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"


#For each stock the data is retrieved and analysis is performed. 
for stock in stocklist:
    print('Processing ', stock)
    try:
        ticker = yfinance.Ticker(stock)
        df = ticker.history(period = historyDuration)
        
        LongEMAIndex = 26
        ShortEMAIndex = 12
        ShortEMAValue = 0.0
        LongEMAValue = 0.0
        PrevShortEMAValue = 0.0
        PrevLongEMAValue = 0.0
        K = 1.0
        SlowMACDIndex = 9
        SlowMACDValue = 0.0
        PrevSlowMACDValue = 0.0
        PrevMACDValue = 0.0
        prevMACDHist = 0.0

        
        CloseSlope = 0.0
        ShortEMASlope = 0.0
        LongEMASlope = 0.0
        MACDHistSlope = 0.0
        SlowMACDSlope = 0.0

        
        for i in range (0,df.shape[0]):
            idx = df.index[i]
            current = df.iloc[i,:]
            if i > 1:
                prev = df.iloc[i-1,:]
                df.loc[idx,'CloseSlope'] = current['Close'] - prev['Close']
            if i < (ShortEMAIndex - 1):
                ShortEMAValue +=  current['Close']
                df.loc[idx, '12EMA'] = 0
            if i < (LongEMAIndex - 1):
                LongEMAValue +=  current['Close']
                df.loc[idx, '26EMA'] = 0
            if i == (ShortEMAIndex - 1):
                ShortEMAValue +=  current['Close']
                ShortEMAValue = ShortEMAValue / ShortEMAIndex
                df.loc[idx, '12EMA'] = ShortEMAValue
                #print(i, ShortEMAValue, PrevShortEMAValue, K)
                PrevShortEMAValue = ShortEMAValue
            if i > (ShortEMAIndex - 1):
                K = 2 / (ShortEMAIndex + 1)
                ShortEMAValue = current['Close'] * K + PrevShortEMAValue * (1 - K)
                df.loc[idx, '12EMA'] = ShortEMAValue
                df.loc[idx, 'ShortEMASlope'] = ShortEMAValue - PrevShortEMAValue
                #print(i, ShortEMAValue, PrevShortEMAValue, K)
                PrevShortEMAValue = ShortEMAValue
            if i == (LongEMAIndex - 1):
                LongEMAValue +=  current['Close']
                LongEMAValue = LongEMAValue / LongEMAIndex
                df.loc[idx, '26EMA'] = LongEMAValue
                PrevLongEMAValue = LongEMAValue
                df.loc[idx, 'MACDHist'] = LongEMAValue - ShortEMAValue
                SlowMACDValue += LongEMAValue - ShortEMAValue
                prevMACDHist = LongEMAValue - ShortEMAValue
            if i > (LongEMAIndex - 1):
                K = 2 / (LongEMAIndex + 1)
                LongEMAValue = current['Close'] * K + PrevLongEMAValue * (1 - K)
                df.loc[idx, '26EMA'] = LongEMAValue
                df.loc[idx, 'LongEMASlope'] = LongEMAValue - PrevLongEMAValue
                df.loc[idx, 'MACDHist'] = ShortEMAValue - LongEMAValue
                df.loc[idx, 'MACDHistSlope'] = (ShortEMAValue - LongEMAValue) - prevMACDHist
                prevMACDHist = ShortEMAValue - LongEMAValue
                if i < (LongEMAIndex + SlowMACDIndex - 1): 
                    SlowMACDValue += LongEMAValue - ShortEMAValue
                    df.loc[idx, 'SignalMACD'] = 0
                elif i == (LongEMAIndex + SlowMACDIndex - 1):
                    SlowMACDValue = SlowMACDValue/SlowMACDIndex
                    df.loc[idx, 'SignalMACD'] = SlowMACDValue
                    PrevSlowMACDValue = SlowMACDValue
                else:
                    K = 2 / (SlowMACDIndex + 1)
                    SlowMACDValue = (LongEMAValue - ShortEMAValue)* K + PrevSlowMACDValue * (1 - K)
                    df.loc[idx, 'SignalMACD'] = SlowMACDValue
                    df.loc[idx, 'SignalMACDSlope'] = SlowMACDValue - PrevSlowMACDValue
                    PrevSlowMACDValue = SlowMACDValue
                PrevLongEMAValue = LongEMAValue
            
            
                
        lastRow = df.shape[0]
        currentStockValues = df.iloc[lastRow-1:]
        positiveIndicator = 0
        
        if currentStockValues['CloseSlope'][0] > 0:
            positiveIndicator += 1
        if currentStockValues['ShortEMASlope'][0] > 0:
            positiveIndicator += 1
        if currentStockValues['LongEMASlope'][0] > 0:
            positiveIndicator += 1
        if currentStockValues['MACDHistSlope'][0] > 0:
            positiveIndicator += 1
        if currentStockValues['SignalMACDSlope'][0] > 0:
            positiveIndicator += 1
        
        analysisResult = analysisResult.append( {'Stock' : stock, 'Close' : currentStockValues['Close'][0], '12EMA': currentStockValues['12EMA'][0], '26EMA': currentStockValues['26EMA'][0], 'MACDHist': currentStockValues['MACDHist'][0], 'SignalMACD': currentStockValues['SignalMACD'][0], 'CloseSlope': currentStockValues['CloseSlope'][0], 'ShortEMASlope': currentStockValues['ShortEMASlope'][0], 'LongEMASlope': currentStockValues['LongEMASlope'][0],  'MACDHistSlope': currentStockValues['MACDHistSlope'][0], 'SignalMACDSlope': currentStockValues['SignalMACDSlope'][0], 'PositiveIndicator' : positiveIndicator}, ignore_index = True)
    except:
        print("Oops!", sys.exc_info()[0], "occurred while processing for stock ", stock, ". Proceeding to next stock")  

    df.to_csv("./" + folderExt + "/" +stock+str(fileExt) + ".csv")
analysisResult.to_csv("AnalysisResult" + str(fileExt) + ".csv")