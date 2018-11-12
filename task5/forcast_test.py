import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

class ForcastTest():
	def __init__(self):
		self.df = []
		self.df1 = []
		#plt.style.use('fivethirtyeight')

	def read_csv(self):
		data = pd.read_csv("./rraman2.csv",header=None) ##change file name here
		data.columns = ['x']
		self.df = data[:1500]
		self.df1 = data[1500:]

	def sma(self):
		best_k = None
		min_rmse = None
		rmse_list = np.zeros(int(len(self.df) / 20 - 2))
		#for n in range(3,4):
		for n in range(3, int(len(self.df) / 20 + 1)): ##start from 3 observations and go till total no of observations/20 + 1
			mva = self.df['x'].rolling(window=n).mean()
			#print(mva.head(10))
			mse = mean_squared_error(self.df['x'][n-1:], mva[n-1:])
			rmse = sqrt(mse)
			#print("RMSE:",rmse)
			rmse_list[n-3] = rmse
		#print(rmse_list)
		best_k = np.argmin(rmse_list) + 3  ##get index of the minimum mean to obtain best k value
		min_rmse = np.min(rmse_list)   ##get minimum of root mean square value
		print("Best Min RMSE,K:",min_rmse,best_k)
		##Ploting RMSE against K values
		plt.plot([3+i for i in range(0,len(rmse_list))],rmse_list)
		plt.title('RMSE vs K')
		plt.xlabel('K')
		plt.ylabel('RMSE')
		plt.savefig('./rmse_k')
		plt.clf()
		return best_k,min_rmse
	
	def plotmva(self,best_k):
		mva = self.df1['x'].rolling(window=best_k).mean()
		## Test accuracy by taking RMSE
		mse = mean_squared_error(self.df1['x'][best_k-1:], mva[best_k-1:])
		rmse = sqrt(mse)
		print("RMSE for Moving Average Model Forecasted Values:",rmse)
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df[best_k-1:],label='Trained Data',color="red")
		plt.plot(self.df1[best_k-1:],label='Test Data',color="orange")
		plt.plot(mva[best_k-1:],label='Forecasted Values using Simple Moving Average',color="black")
		plt.ylim(-10,0)
		plt.title('Simple Moving Average Plot')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./mv_avg_plot')
		plt.clf()

	def smoothing(self):
		alist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
		rmse_list = []
		for alpha in alist:
			savg = [self.df['x'][0]]
			savgm = SimpleExpSmoothing(self.df)
			savgfit = savgm.fit(alpha,optimized=False)
			savg = savgfit.fittedvalues
			#print(savg)
			mse = mean_squared_error(self.df['x'], savg)
			rmse = sqrt(mse)
			rmse_list.append(rmse)
		#print(rmse_list)
		best_a = alist[np.argmin(rmse_list)]  ##get index of the minimum mean to obtain best alpha value
		min_rmse = np.min(rmse_list)   ##get minimum of root mean square value
		print("Best Min RMSE,Alpha using fit:",min_rmse,best_a)


		##Ploting RMSE against K values
		plt.plot(alist,rmse_list)
		plt.title('RMSE vs alpha')
		plt.xlabel('a')
		plt.ylabel('RMSE')
		plt.savefig('./rmse_a')
		plt.clf()
		return best_a,min_rmse
	
	def plotsmooth(self,alpha):
		savg = [self.df['x'][0]]
		savgm = SimpleExpSmoothing(self.df)
		savgfit = savgm.fit(alpha,optimized=False)
		savg = savgfit.forecast(len(self.df1))
		## Test accuracy by taking RMSE
		mse = mean_squared_error(self.df1['x'], savg)
		rmse = sqrt(mse)
		print("RMSE for Simple Exponential Smoothing Model Forecasted Values:",rmse)
		#print(savg)
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df['x'],label='Trained Data',color="red")
		plt.plot(self.df1['x'],label='Test Data',color="orange")
		plt.plot(savg,label='Forecasted Values Using Moving Average',color="black")
		plt.ylim(-10,0)
		plt.title('Exponential Smoothing Model')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./sm_avg_plot')
		plt.clf()

	def bothplot(self):
		pacflag = pacf(self.df,nlags=30)
		##plot pacf
		fig = plt.figure(figsize=(16,8))
		plt.plot(pacflag)
		plt.axhline(y=0,color='gray')
		plt.axhline(y=-1.96/np.sqrt(len(self.df)),color='blue')
		plt.axhline(y=1.96/np.sqrt(len(self.df)),color='blue')
		plt.title('Partial Autocorrelation for the DataSet')
		plt.savefig('./pacf_plot')
		plt.clf()

	def armodel(self,p,d,q):
		self.model = ARIMA(self.df, order=(p, d, q))
		arfit = self.model.fit(disp=0)
		print(arfit.summary())
		arforecasted = arfit.predict(start=len(self.df),end=len(self.df)+len(self.df1)-1,dynamic=True)
		#arforecasted = arfit.forecast(len(self.df1))
		#print(arforecasted)
		## Test accuracy by taking RMSE
		mse = mean_squared_error(self.df1['x'], arforecasted)
		rmse = sqrt(mse)
		print("RMSE for AR(1) Model Forecasted Values:",rmse)
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df['x'],label='Original Values',color="red")
		plt.plot(self.df1['x'],label='Test Data',color="orange")
		plt.plot(arforecasted,label='Predicted Values Using AR(1) model',color="black")
		plt.ylim(-10,1)
		plt.title('AR(1) Model')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./arima_fittedplot')
		plt.clf()

testf = ForcastTest()
testf.read_csv()
best_k,min_rmse = testf.sma()
testf.plotmva(best_k)
best_a,min_rmse = testf.smoothing()
testf.plotsmooth(best_a)
testf.bothplot()
testf.armodel(1,0,0)
