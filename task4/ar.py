import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm
import matplotlib.mlab as mlab
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import chisquare

class AR():
	def __init__(self):
		self.df = []
		self.model = None
		#plt.style.use('fivethirtyeight')

	def read_csv(self):
		data = pd.read_csv("./rraman2.csv",header=None) ##change file name here
		data.columns = ['x']
		self.df = data

	def pacf(self):
		fig = plt.figure(figsize=(16,8))
		plot_pacf(self.df, lags=30)
		plt.title('PCAF Plot for Dataset')
		plt.savefig('./pcaf_plot_tsaplots')
		plt.clf()
	
	def bothplot(self):
		acflag = acf(self.df,nlags=30)
		pacflag = pacf(self.df,nlags=30)
		##plot pacf
		fig = plt.figure(figsize=(16,8))
		plt.plot(pacflag)
		plt.axhline(y=0,color='gray')
		plt.axhline(y=-1.96/np.sqrt(len(self.df)),color='blue') ##confidence interval
		plt.axhline(y=1.96/np.sqrt(len(self.df)),color='blue')
		plt.title('Partial Autocorrelation for the DataSet')
		plt.savefig('./pacf_plot')
		plt.clf()

	def armodel(self,p,d,q):
		self.model = ARIMA(self.df, order=(p, d, q))
		arfit = self.model.fit(disp=0)
		print(arfit.summary())
		arfitted = arfit.fittedvalues
		mse = mean_squared_error(self.df['x'], arfitted)
		rmse = sqrt(mse)
		print("RMSE:",rmse)
		fig = plt.figure(figsize=(16,8))
		plt.plot(self.df['x'],label='Original Values',color="red")
		plt.plot(arfitted,label='Predicted Values Using AR(1) model',color="black")
		plt.ylim(-10,0)
		plt.title('AR(1) Model')
		plt.legend(loc='best',fancybox=True,ncol=2)
		plt.savefig('./arima_fittedplot')
		plt.clf()
		return arfit

	def get_residuals(self,modelfit):
		res = modelfit.resid #fetching residuals from the model
		return res

	def qqplot(self,res):
		qq = sm.qqplot(res, stats.t, fit=True, line='45') ##qq plot for residuals
		plt.savefig('./qqplot_x')
		plt.clf()

	def histogram(self,res):
		(mu, sigma) = norm.fit(res)
		n, bins, patches = plt.hist(res,normed=1, bins='auto', align='mid', color='navy',edgecolor='white')
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.savefig('./histo_x')
		plt.clf()
	
	def s_plot(self,x,res,i):
		s_plot = sns.residplot(x[["x1"]],res)
		plt.xlabel('x1')
		plt.ylabel('y')
		plt.savefig('./scatter_x1'+str(i))
		plt.clf()

	def s_plot(self,res):
		s_plot = sns.residplot(self.df['x'],res)
		plt.xlabel('x')
		plt.ylabel('residual')
		plt.savefig('./scatter_x')
		plt.clf()

	def chisquare(self,res):
		chi_s = chisquare(res)
		print("Chi-square Results on the Residuals:",chi_s)

		

ar = AR()
ar.read_csv()
ar.pacf()
ar.bothplot()
modelfit = ar.armodel(1,0,0) ##using p=1 from the pacf plots and d and q to be 0 since AR(1) model
res = ar.get_residuals(modelfit)
ar.qqplot(res)
ar.histogram(res)
ar.s_plot(res)
ar.chisquare(res)

	
