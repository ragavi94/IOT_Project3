import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

class Exponential_Smoothing():
	def __init__(self):
		self.df = []
		self.model = None
		#plt.style.use('fivethirtyeight')

	def read_csv(self):
		data = pd.read_csv("./rraman2.csv",header=None) ##change file name here
		data.columns = ['x']
		self.df = data

	def smoothing(self):
		alist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
		#alist = [0.1]
		rmse_list = []
		'''for alpha in alist:
			savg = [self.df['x'][0]]
			for n in range(1, len(self.df)):
				savg.append(alpha * self.df['x'][n-1] + (1 - alpha) * savg[n-1])
			mse = mean_squared_error(self.df['x'], savg)
			rmse = sqrt(mse)
			rmse_list.append(rmse)
		print(rmse_list)
		best_a = alist[np.argmin(rmse_list)]  ##get index of the minimum mean to obtain best alpha value
		min_rmse = np.min(rmse_list)   ##get minimum of root mean square value
		print("Best Min RMSE,Alpha:",min_rmse,best_a)'''
		##another method here	
		
		for alpha in alist:
			savg = [self.df['x'][0]]
			savgm = SimpleExpSmoothing(self.df)
			savgfit = savgm.fit(alpha,optimized=False)
			savg = savgfit.fittedvalues
			#print(savg)
			mse = mean_squared_error(self.df['x'], savg)
			rmse = sqrt(mse)
			#print("RMSE for alpha=0.1:",rmse)
			rmse_list.append(rmse)
		print(rmse_list)
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
	
	def plot(self,alpha):
		savg = [self.df['x'][0]]
		savgm = SimpleExpSmoothing(self.df)
		savgfit = savgm.fit(alpha,optimized=False)
		savg = savgfit.fittedvalues
		#print(savg)
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df['x'],label='Original Values',color="red")
		plt.plot(savg,label='Predicted Values Using Exponential Smoothing',color="black")
		plt.ylim(-10,0)
		plt.title('Exponential Smoothing Model')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./sm_avg_plot')
		plt.clf()
					

es = Exponential_Smoothing()
es.read_csv()
best_a,min_rmse = es.smoothing()
es.plot(best_a)
