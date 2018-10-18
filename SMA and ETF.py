

### Overall, quite a decent job, Gaurav. Score 17/20
### Please go through the comments for each question.###
### I've also provided the solutions, go through it for learning about different approaches ###


import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import os
import talib as ta


####Question 1

#help(ta)
print(ta.get_function_groups())

os.chdir('E:\Quantinsti\EPAT 06 - ATP\ATP 2')

df = pd.read_csv("MSFT.csv", index_col='Date', parse_dates = True)
df2 = df

open = []
high = []
low = []
close = []

open = df2['Open']
high = df2['High']
low = df2['Low']
close = df2['Close']

df2['ema']= ta.EMA(np.array(close),timeperiod = 10)

df2['MACD'],df2['MACDSignal'],df2['MACDHist'] = ta.MACD(np.array(close), fastperiod=12, slowperiod = 26, signalperiod = 9)

df2['3white'] = ta.CDL3WHITESOLDIERS(np.array(open),np.array(high),np.array(low),np.array(close))

################################################################
###Question 2

# We Buy when Price > 10-month SMA
# We Sell and move to cash when Price < 10-month SMA

buyPrice = 0.0
sellPrice = 0.0
maWealth = 1.0
cash = 1
stock = 0
sma = 200

ma = np.round(df2['Close'].rolling(window=sma, center=False).mean(), 2)

n_days = len(df2['Close'])

closePrices = df2['Close']

buy_data = []

sell_data = []

trade_price = []

wealth = []

def signal(buyPrice, sellPrice, maWealth, cash, stock, sma):
    for d in range(sma-1, n_days):
        # buy if Stockprice > MA & if not bought yet
        if closePrices[d] > ma[d] and cash == 1:
            buyPrice = closePrices[d + 1]
          
            buy_data.append(buyPrice)
            trade_price.append(buyPrice)
            cash = 0
            stock = 1
        
        # sell if Stockprice < MA and if you have a stock to sell  
        if closePrices[d] < ma[d]  and stock == 1:
            sellPrice = closePrices[d + 1]
          
            cash = 1
            stock = 0
            sell_data.append(sellPrice)
            trade_price.append(sellPrice)
            maWealth = maWealth * (sellPrice / buyPrice)
            wealth.append(maWealth)

#This line always hangs in my system, however this should give the desired result.    
#df2['Close'].apply(lambda x: signal(buyPrice, sellPrice, maWealth, cash, stock, sma))

w = pd.DataFrame(wealth)

plt.plot(w)


### Multiple issues here. First, when a function is called a local namespace is created
### So you won't be able to access trade_price, maWealth. Updates to wealth will not be visible outside
### the function. Second, you've used the for loop. Using .apply would be the way to go.
### Refer to the solutions provided.
### Score 2/5            

################################################################
###Question 3

# We Buy when Price > 10-month SMA
# We Sell and move to cash when Price < 10-month SMA

buyPrice = 0.0
sellPrice = 0.0
maWealth = 1.0
cash = 1
stock = 0
#creating a list of different moving averages
sma = [50,100,150,200]
ma=[[] for i in range(len(sma))]
for i in range(0,len(sma)):
    ma[i] = np.round(df2['Close'].rolling(window=sma[i], center=False).mean(), 2)
    
    
n_days = len(df2['Close'])

closePrices = df2['Close']

buy_data = []

sell_data = []

trade_price = []

wealth = []

#creating wealth and tradeprice list the number of sma periods
tp=[[] for i in range(len(sma)+1)]
w=[[] for i in range(len(sma)+1)]


#running a for loop for all sma's
for n in range(0,len(sma)):
    
            
    buyPrice = 0.0
    sellPrice = 0.0
    maWealth = 1.0
    cash = 1
    stock = 0
    buy_data = []
    sell_data = []
    trade_price = []
    wealth = []
    
    for d in range(sma[n]-1, n_days):
        # buy if Stockprice > MA & if not bought yet
        if closePrices[d] > ma[n][d] and cash == 1:
            buyPrice = closePrices[d + 1]
          
            buy_data.append(buyPrice)
            trade_price.append(buyPrice)
            cash = 0
            stock = 1
        
        # sell if Stockprice < MA and if you have a stock to sell  
        if closePrices[d] < ma[n][d]  and stock == 1:
            sellPrice = closePrices[d + 1]
            cash = 1
            stock = 0
            sell_data.append(sellPrice)
            trade_price.append(sellPrice)
            maWealth = maWealth * (sellPrice / buyPrice)
            wealth.append(maWealth)   
            w[n]= (wealth)    
    tp[n] = trade_price
    
df2['Buy & Hold Returns'] = np.log(df2['Close'] / df2['Close'].shift(1))

#printing returns for different moving averages
print("Cumulative returns, Buy and Hold"+str(df2['Buy & Hold Returns'].sum())+
                                             "\nWealth at "+str(sma[0])+"MA "+str(sum(w[0]))+
                                             "\nWealth at "+str(sma[1])+"MA "+str(sum(w[1]))+
                                             "\nWealth at "+str(sma[2])+"MA "+str(sum(w[2]))+
                                             "\nWealth at "+str(sma[3])+"MA "+str(sum(w[3])))

### Good. ###
### Score 5/5 ###

											 
################################################################
###Question 4

##### Strategy II: The Moving Average Crossover Strategy ######
##### Close Price is used to do the calculations

shortPeriod = 20
longPeriod = 40

df2['shortMA'] = df2['Close'].rolling(window=shortPeriod, center=False).mean()
df2['longMA'] = df2['Close'].rolling(window=longPeriod, center=False).mean()

# WHY DOUBLE BRACKET
### When you select multiple columns to plot graphs, you enclose it as a list ###
df2[['Close','shortMA','longMA']].plot(grid=False, linewidth=0.8) 

df2['shortMA2'] = df2['Close'].rolling(window=shortPeriod, center=False).mean().shift(1)
df2['longMA2'] = df2['Close'].rolling(window=longPeriod, center=False).mean().shift(1)

#Generating Signal

df2['Signal'] = np.where((df2['shortMA']>df2['longMA'])
                         & (df2['shortMA2']<df2['longMA2']), 1, 0)
df2['Signal'] = np.where((df2['shortMA']<df2['longMA'])
                         & (df2['shortMA2']>df2['longMA2']), -1, df2['Signal'])

df2['Buy'] = df2.apply(lambda x: x['Close'] if x['shortMA'] > x['longMA'] 
                        and x['shortMA2'] < x['longMA2'] else 0, axis=1)

df2['Sell'] = df2.apply(lambda y: -y['Close'] if y['shortMA'] < y['longMA'] 
                        and y['shortMA2'] > y['longMA2'] else 0, axis=1)


df2['TP'] = df2['Buy'] + df2['Sell']
df2['TP']
df2['TP']=df2['TP'].replace(to_replace=0, method='ffill')


df2['Position'] = df2['Signal'].replace(to_replace=0, method= 'ffill')
#Generating new Positions with NO SHORTING rule
df2['Position2'] = df2['Signal'].replace(to_replace=0, method= 'ffill')
#replacing all shorting signals with null
df2['Position2'] = df2['Position2'].replace([-1],[0] )

k = df2['Buy'].nonzero()

len(k[0]) # total number of trades
df2['Position'].plot(grid=True, linewidth=1)

df2['Buy & Hold Returns'] = np.log(df2['Close'] / df2['Close'].shift(1))
df2['Strategy Returns'] = df2['Buy & Hold Returns'] * df2['Position'].shift(1)
df2['No Short Strategy Returns'] = df2['Buy & Hold Returns'] * df2['Position2'].shift(1)

#Plotting returns from multiple strategies
df2[['Buy & Hold Returns', 'Strategy Returns', 'No Short Strategy Returns']].cumsum().plot(grid=True, figsize=(9,5))

### Correct. Score 5/5. Good.


###############################################################
#####Question 5

stocks = ["AXISBANK.NS", "BANKBARODA.NS", "BHEL.NS","BPCL.NS","BHARTIARTL.NS"]


end = datetime.datetime.now().date()

start = end - pd.Timedelta(days=365*5)

f = web.get_data_yahoo(stocks, start, end)

#Storing data and creating dataframes
df.to_pickle('f.pickle')
df_open = f.loc["Open"].copy() 
df_close = f.loc["Close"].copy()
df_close = df_close.shift(1)
df_returns = df_close.divide(df_open, axis='index') - 1

#Saving returns file
df_returns.to_csv("out.csv")
#dropping nan values
df_returns = df_returns.dropna()
df_returns = df_returns.applymap(float)
df_returns.loc['cumReturns'] = 0

#Cumulative Returns added at the end of dataframe 'df_returns'
for i in range(len(df_returns.columns.values)):
    df_returns.loc["cumReturns"][df_returns.columns.values[i]] = pd.Series(df_returns[df_returns.columns.values[i]].sum(), index = [df_returns.columns.values[i]])

#defigning variables for hit ratio calculation
pos_count = 0
neg_count = 0
df_returns.loc['hitRatio'] = 0
df_returns = df_returns.dropna()
df_returns = df_returns.applymap(float)

for i in range(len(df_returns.columns.values)):
    for j in range(len(df_returns)-1):
        if df_returns.iloc[j,i]>0:
            pos_count += 1
        if df_returns.iloc[j,i]<0:
            neg_count += 1
#hit ratio added at the end of the dataframe returns
    df_returns.loc["hitRatio"][df_returns.columns.values[i]] = pd.Series(pos_count/neg_count, index = [df_returns.columns.values[i]])
    pos_count = 0
    neg_count = 0

#Creating list of the cumulative returns and hit ratio
temp1 = []
temp2 = []
for i in range(len(df_returns.columns.values)):
    temp1.append(df_returns.ix['cumReturns', i])
    temp2.append(df_returns.ix['hitRatio', i])

#plotting the two graphs
plt.plot(temp1)
plt.plot(temp2)

################################################################

### Correct. Well done. Score 5/5

###Question 6

# We Buy when Price > 10-month SMA
# We Sell and move to cash when Price < 10-month SMA

buyPrice = 0.0
sellPrice = 0.0
maWealth = 1.0
cash = 1
stock = 0
ema = 20

ma = ta.EMA(np.array(df2['Close']), timeperiod = ema)
ma = np.array(ma)
n_days = len(df2['Close'])

closePrices = df2['Close']

buy_data = []

sell_data = []

trade_price = []

wealth2 = []

for d in range(ema-1, n_days):
    # buy if Stockprice > MA & if not bought yet
    if closePrices[d] > ma[d] and cash == 1:
        buyPrice = closePrices[d + 1]
      
        buy_data.append(buyPrice)
        trade_price.append(buyPrice)
        cash = 0
        stock = 1
    
    # sell if Stockprice < MA and if you have a stock to sell  
    if closePrices[d] < ma[d]  and stock == 1:
        sellPrice = closePrices[d + 1]
      
        cash = 1
        stock = 0
        sell_data.append(sellPrice)
        trade_price.append(sellPrice)
        maWealth = maWealth * (sellPrice / buyPrice)
        wealth2.append(maWealth)

w1 = pd.DataFrame(wealth2)
#Plot for EMA, if the line in Question 2 works, wealth for SMA can be added in this chart as well.
plt.plot(w1)


### Same issues as Q1.
### Score 3/5

