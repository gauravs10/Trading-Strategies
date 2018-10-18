import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Importing data
data = pd.read_csv('QDA.csv')
data.info()
data.describe()
# Copy of data
es = data.copy()

# Creating a delta column
# Deltha is the difference in bid volume an ask volume, acting as a measure of the momentum
es['Delta'] = es[' BidVolume']-es[' AskVolume']

# Creating a regression plot of the Delta and the Closing price

#sns.lmplot(x='Delta', y=' Close', data = es)
plt.title('ES Regression Delta to Close')

# Creating a distribution of delta
#sns.distplot(es['Delta'])

# adding a range to our data
es['Range'] = es[' High'] - es[' Low']
#sns.jointplot(es['Delta'], es['Range'])

# Distribution of number of trades per bar
#sns.distplot(es[' NumberOfTrades'], bins=100)

class rsi_strategy(object):
    def __init__(self,data,n,data_name,start,end):
        
        self.data = data
        self.n = n
        self.data_name = data_name
        self.start = start
        self.end = end
        
    def generate_signals(self):
    
        delta = self.data[' Close'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp<0] = 0
        dDown[dDown>0] = 0
        RolUp = dUp.rolling(self.n).mean()
        RolDown = dDown.rolling(self.n).mean()
        
        # assigning indicator to the Dataframe
        self.data['RSI'] = np.where(RolDown!=0, RolUp/RolDown,1)
        self.data['RSI_Slow'] = self.data['RSI'].rolling(self.n).mean()
        
        self.data = self.data.assign(Signal = pd.Series(np.zeros(len(self.data))).values)
        self.data.loc[self.data['RSI']<self.data['RSI_Slow'], 'Signal'] = 1
        self.data.loc[self.data['RSI']>self.data['RSI_Slow'], 'Signal'] = -1
                
    def plot_performance(self, allocation):
        self.allocation = allocation
        
        # Creating returns and portfolio value series
        self.data['Return']=np.log(self.data[' Close']/self.data[' Close'].shift(1))
        self.data['S_Return']=self.data['Signal'].shift(1)*self.data['Return']
        self.data['Market_Return']=self.data['Return'].expanding().sum()
        self.data['Strategy_Return']=self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value']=((self.data['Strategy_Return']+1)*self.allocation)                   
            
         #creating metrics
        self.data['Wins']=np.where(self.data['S_Return'] > 0,1,0)
        self.data['Losses']=np.where(self.data['S_Return']<0,1,0)
        self.data['Total Wins']=self.data['Wins'].sum()
        self.data['Total Losses']=self.data['Losses'].sum()
        self.data['Total Trades']=self.data['Total Wins'][0]+self.data['Total Losses'][0]
        self.data['Hit Ratio']=round(self.data['Total Wins']/self.data['Total Losses'],2)
        self.data['Win Pct']=round(self.data['Total Wins']/self.data['Total Trades'],2)
        self.data['Loss Pct']=round(self.data['Total Losses']/self.data['Total Trades'],2)
        
        #Plotting the Performance of the RSI Strategy
        plt.plot(self.data['Market_Return'],color='black', label='Market Returns')
        plt.plot(self.data['Strategy_Return'],color='blue', label= 'Strategy Returns')
        plt.title('%s RSI Strategy Backtest'%(self.data_name))
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
         
        plt.plot(self.data['Portfolio Value'])
        plt.title('%s Portfolio Value'%(self.data_name))
        plt.show()    
            
        
# Creating an instance of Strategy Class
strat1 = rsi_strategy(es, 10, 'ES',es['Date'][0], es['Date'].iloc[-1])
# Generating Signals
strat1.generate_signals()
# Plotting Performance
strat1.plot_performance(10000)
