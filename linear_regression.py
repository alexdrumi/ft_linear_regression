#!/opt/homebrew/bin/python3

import pandas as pd
import numpy as np

'''
#!/opt/homebrew/bin/python3
#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''


class LinearRegression:

	def __init__(self,learning_rate=1e-2, convergence_treshold=1e-6, max_iterations=6000):
		self.df = None
		self.filename = 'training_data.csv'

		self.theta0 = 0 #intercept
		self.theta1 = 0 #slope
		
		self.learning_rate = learning_rate
		self.convergence_threshold = convergence_treshold
		self.max_iterations = max_iterations
		self.iterations = 0
		self.initial_treshold = 0.0001

		#gradients
		self.derivative_intercept = 0
		self.derivative_slope = 0
		self.step_size_intercept = 0
		self.step_size_slope = 0
		
		#data
		self.mileage = 0
		self.price = 0

		#mse history
		self.mse_history = [self.compute_MSE()]



	def compute_MSE(self):
		predictions = self.theta0 + self.theta1 * self.mileage #theta0 = intercept + slope * mileage
		return np.mean((self.price - predictions) ** 2)



	def compute_RMSE(self):
		return np.sqrt(self.compute_MSE())



	def compute_MAE(self):
		predictions = self.theta0 + self.theta1 * self.mileage
		mae = np.mean(np.abs(self.price - predictions))
		return mae


	'''
	------------------------------------------------------------------------------
	The following is calculating the derivatives respect to slope and intercept
	------------------------------------------------------------------------------


	------------------------------------------------------------------------------
	SSR = (observed_price - (intercept + 0.5 * observed_mileage)) ** 2


	derivative of SSR with respect to intercept:

	SSR = (observed_price - (intercept + 0.5 * observed_mileage))^2
	y = observed_price
	b = intercept
	x = observed_mileage

	SSR = (y - (b + 0.5x))^2

	h = y-(b+0.5x)
	h'(b) = -1
	#we need the derivative of h respect to b (intercept)
	#dh/db = d/db[y] - d/db[b+0.5x] = 0 + (-1 + 0) = -1

	g = h^2
	g'(h) = 2h

	g'(u) * h'(b) = 2u * -1
	= 2(y-(b+0.5x)) * -1
	= -2(y-(b+0.5x)) 

	dSSR/dIntercept = -2(observed_price-(intercept + 0.5 * observed_mileage))
	------------------------------------------------------------------------------


	------------------------------------------------------------------------------
	derivative of SSR with respect to slope:

	SSR = (y - (b + c*x))^2
	y = observed_price
	b = intercept
	x = observed_mileage
	c = slope

	h = y - (b + c*x)
	dh/dc = d/dc[y] - d/dc[b + cx] = 0 - (0 + x) = -x
	g = h^2
	g'(h) = 2h

	g'(u) * h'(b) = 2u * -x
	= 2(y-(b+c*x)) * -x
	= -2(y-(b+c*x)) * -x
	= -2x(y-(b+c*x))

	divide the whole thing with the amount of datapoints to get the MSE
	------------------------------------------------------------------------------
	'''
	# def compute_MSE_gradients(self):
	# 	x, y = self.mileage,self.price
	# 	b = self.theta0 #intercept
	# 	c = self.theta1 #slope
	# 	n = len(self.price) #m in the subject

	# 	#calculating residuals
	# 	h = y - (b + c * x)

	# 	#derivative of h respect to intercept
	# 	# dh_db = -1
	# 	#derivative of SSR respect to intercept, applying chain rule (this is 1/m in the subject)
	# 	# dMSE_db = (2/n) * np.sum(h * dh_db)
  
	# 	#derivative of h respect to slope
	# 	#dh_dc = -x
	# 	#derivative of SSR respect to slope, applying chain rule, divided by n (this is 1/m in the subject)
	#	# dMSE_dc = (2/n) * np.sum(h * dh_dc)
	# 	return (dMSE_db, dMSE_dc)

	def compute_MSE_gradients(self):
		x, y = self.mileage, self.price
		b, c = self.theta0, self.theta1 
		n = len(self.price)

		residuals = y - (b + c * x)
		dMSE_db = -2 * np.mean(residuals) #this is a faster calculation than summing and dividing/multiplying
		dMSE_dc = -2 * np.mean(residuals * x) #this is a faster calculation than summing and dividing/multiplying

		return dMSE_db, dMSE_dc
	


	def min_max_normalize(self, array):
		x_min = np.min(array)
		x_max = np.max(array)
		array_normalized = (array - x_min)/(x_max - x_min)

		return array_normalized



	def convergence_succeeded(self):
		return (abs(self.step_size_intercept) < self.convergence_threshold 
			and abs(self.step_size_slope) < self.convergence_threshold)



	def log_MSE(self):
		mse_current = self.compute_MSE()
		self.mse_history.append(mse_current)



	'''
	Take a look at RMSprop, Adam, or incorporating learning rate schedules.
	'''
	def gradient_descent(self):
		while (self.iterations < self.max_iterations):
			self.derivative_intercept, self.derivative_slope = self.compute_MSE_gradients()

			self.step_size_intercept = self.learning_rate * self.derivative_intercept
			self.step_size_slope = self.learning_rate * self.derivative_slope 

			self.theta0 -= self.step_size_intercept
			self.theta1 -= self.step_size_slope

			if self.max_iterations % 100 == 0:
				self.log_MSE()

			if self.convergence_succeeded() is True:
				break

			self.iterations += 1
		
		return self.theta0, self.theta1

