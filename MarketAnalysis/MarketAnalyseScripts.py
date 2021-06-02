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
analysisResult = pd.DataFrame(columns = ['Stock', 'Close', '12EMA', '26EMA', 'MACDHist', 'SignalMACD', 'CloseSlope', '12EMASlope', '26EMASlope', 'MACDHistSlope', 'SignalMACDSlope', 'PositiveIndicator'])


#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"


#For each stock the data is retrieved and analysis is performed. 
for stock in stocklist:
    print('Processing ', stock)
    try:
        ticker = yfinance.Ticker(stock)
        df = ticker.history(period = historyDuration)

        df['12EMA'] = df.Close.ewm(span=12, adjust=False).mean()
        df['26EMA'] = df.Close.ewm(span=12, adjust=False).mean()
        df['MACDHist'] = df['12EMA'] - df['26EMA'] 
        df['SignalMACD'] = df['MACDHist'].ewm(span=9, adjust=False).mean()

        df['CloseSlope'] = df.Close.diff()
        df['12EMASlope'] = df['12EMA'].diff()
        df['26EMASlope'] = df['26EMA'].diff()
        df['MACDHistSlope'] = df['MACDHist'].diff()
        df['SignalMACDSlope'] = df['SignalMACD'].diff()
        df['PositiveIndicator'] = 0
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['CloseSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['12EMASlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['26EMASlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['MACDHistSlope'].apply(lambda x: 1 if x > 0 else 0)
        df['PositiveIndicator'] = df['PositiveIndicator'] + df['SignalMACDSlope'].apply(lambda x: 1 if x > 0 else 0)

        lastRow = df.shape[0]
        currentStockValues = df.iloc[lastRow-1:]

        analysisResult = analysisResult.append( {'Stock' : stock, 'Close' : currentStockValues['Close'][0], '12EMA': currentStockValues['12EMA'][0], '26EMA': currentStockValues['26EMA'][0], 'MACDHist': currentStockValues['MACDHist'][0], 'SignalMACD': currentStockValues['SignalMACD'][0], 'CloseSlope': currentStockValues['CloseSlope'][0], '12EMASlope': currentStockValues['12EMASlope'][0], '26EMASlope': currentStockValues['26EMASlope'][0],  'MACDHistSlope': currentStockValues['MACDHistSlope'][0], 'SignalMACDSlope': currentStockValues['SignalMACDSlope'][0], 'PositiveIndicator' : currentStockValues['PositiveIndicator'][0]}, ignore_index = True)
    except:
        print("Oops!", sys.exc_info()[0], "occurred while processing for stock ", stock, ". Proceeding to next stock")  

    df.to_csv("./" + folderExt + "/" +stock+str(fileExt) + ".csv")
analysisResult.to_csv("AnalysisResult" + str(fileExt) + ".csv")