import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
import matplotlib.mlab as mlab


class Check_Stationary:
	def __init__(self):
		self.df = []

	def read_csv(self):
		reader = csv.reader(open('./rraman2.csv','r'),delimiter=',') ##change file name here
		columns = list(zip(*reader))
		self.df = [float(i) for i in columns[0]]

	def plot(self,x):
		fig = plt.figure(1)
		plt.plot(x)
		plt.xlabel('x')
		plt.savefig('./tsplot')
		plt.clf()

	def check_stationary(self,x):
		print("checking for stationary using Dickey-Fuller")
		dftest = adfuller(x[:1500], autolag='AIC')
		#print(dftest)
		dfoutput = pd.Series(dftest[0:4], index=['Dickey-Fuller Test Statistics','p-value','#Lags Used','Number of Observations Used'])
		for k,v in dftest[4].items():
        		dfoutput['Critical Values (%s)'%k] = v
    		print(dfoutput)

	def histogram(self,arr):
		(mu, sigma) = norm.fit(arr)
		n, bins, patches = plt.hist(arr,normed=1, bins='auto', align='mid', color='navy',edgecolor='white')
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.savefig('./histo_x')
		plt.clf()
		


ts = Check_Stationary()
ts.read_csv()
ts.plot(ts.df)
ts.check_stationary(ts.df)
ts.histogram(ts.df)



