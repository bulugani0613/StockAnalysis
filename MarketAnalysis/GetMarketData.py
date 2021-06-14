import sys
import pandas as pd
import yfinance
from datetime import datetime
import os 
from os import path
import numpy as np
import requests


#ScriptList contains the list of script requiring analysis 
stocklistFileHandler = open("GetMarketDataList.txt")
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
analysisResult = pd.DataFrame(columns = ['Stock', 'Close', '12EMA', '26EMA', 'FastMACD', 'SignalMACD', 'MACDHist', 'ADX', 'RSI', 'CloseSlope', '12EMASlope', '26EMASlope', 'FastMACDSlope', 'SignalMACDSlope', 'MACDHistSlope', 'ADXSlope', 'BuyIndicator'])


#For each stock the data is retrieved, analyzed and result stored in the stocknameYYYYMMDDHHMMSS.csv file. Summary stored in analysisResultYYYYMMDDHHMMSS.csv
historyDuration = "5mo"

def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]
    
def getStockData(stock):
    url = "https://priceapi.moneycontrol.com/techCharts/techChartController/history?symbol=" + stock + "&resolution=1D&from=1588996056&to=1623124080"
    #print(url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"})
    #print(url)
    #print(response.json())
    data = response.json()
    df = pd.DataFrame(columns = ["TimeStamp", "Date", "Open", "Close", "High", "Low", "Volume"])
    df['TimeStamp'] = data.get('t')
    df['Open'] = data.get('o')
    df['Close'] = data.get('c')
    df['High'] = data.get('h')
    df['Low'] = data.get('l')
    df['Volume'] = data.get('v')
    df['Date'] = df['TimeStamp'].apply(lambda x: getDate(x))
    df.set_index("Date", inplace = True)
    return df
    
def getDate(timestampint):
    timestamp = datetime.fromtimestamp(timestampint)
    return timestamp.strftime('%Y-%m-%d')

#For each stock the data is retrieved and analysis is performed. 
for stock in stocklist:
    print('Processing ', stock)
    try:
        #ticker = yfinance.Ticker(stock)
        #origdf = ticker.history(period = historyDuration)
        origdf = getStockData(stock)
        #print(origdf)
        df = origdf[['Open']]
        df.columns  = {'Close'}
        df = pd.concat([df, origdf[['Close']]], axis=0, ignore_index=True)

        df.columns  = {'High'}
        df = pd.concat([df, origdf[['High']]], axis=0, ignore_index=True)

        df.columns  = {'Low'}
        df = pd.concat([df, origdf[['Low']]], axis=0, ignore_index=True)


        df_out = pd.concat([origdf[['Open']], origdf[['Close']], origdf[['High']], origdf[['Low']]], axis=0, ignore_index=True)

        result_df = df.Low.apply(lambda x: round(x)).value_counts()
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Oops!", sys.exc_info()[0], "occurred while processing for stock ", stock, ". Proceeding to next stock. Error is in ", exc_type, fname, exc_tb.tb_lineno) 
    result_df.to_csv("./" + folderExt + "/" +stock+str(fileExt) + ".csv")