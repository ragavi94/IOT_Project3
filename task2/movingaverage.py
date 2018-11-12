import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import matplotlib.pyplot as plt

class Moving_Average():
	def __init__(self):
		self.df = []
		self.model = None
		#plt.style.use('fivethirtyeight')

	def read_csv(self):
		data = pd.read_csv("./rraman2.csv",header=None) ##change file name here
		data.columns = ['x']
		self.df = data
	def sma(self):
		best_k = None
		min_rmse = None
		rmse_list = np.zeros(int(len(self.df) / 20 - 2))
		#for n in range(5,6):
		for n in range(3, int(len(self.df) / 20 + 1)): ##start from 3 observations and go till total no of observations/20 + 1
			mva = self.df['x'].rolling(window=n).mean()
			#print(mva.head(10))
			mse = mean_squared_error(self.df['x'][n-1:], mva[n-1:])
			rmse = sqrt(mse)
			#print("RMSE for K=5:",rmse)
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
	
	def plot(self,best_k):
		mva = self.df['x'].rolling(window=best_k).mean()
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df[best_k-1:],label='Original Values',color="red")
		plt.plot(mva[best_k-1:],label='Predicted Values Using Moving Average',color="black")
		plt.ylim(-10,0)
		plt.title('Simple Moving Average Plot')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./mv_avg_plot')
		plt.clf()
	
ma = Moving_Average()
ma.read_csv()
best_k,min_rmse = ma.sma()
ma.plot(best_k)


