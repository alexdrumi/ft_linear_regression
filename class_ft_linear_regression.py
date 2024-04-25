#!/opt/homebrew/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
#!/opt/homebrew/bin/python3
#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''



class LinearRegression:

	def __init__(self, df):
		self.df = df
		self.theta0 = 0
		self.theta1 = 0



	def calculate_least_squares_values(self, df):
		n = df.shape[0]
		# n = 3
		xy = df['km'] * df['price']
		x_squared = df['km'] * df['km']

		x_sum = df['km'].sum()
		y_sum = df['price'].sum()
		xy_sum = xy.sum()
		x_squared_sum = x_squared.sum()

		return n, x_sum, y_sum, xy_sum, x_squared_sum



	def calculate_slope(self, n, x_sum, y_sum, xy_sum, x_squared_sum):
		m_nominator = (n * xy_sum) - (x_sum * y_sum)
		m_denominator = (n * x_squared_sum) - (x_sum ** 2)
		m = m_nominator / m_denominator

		return m



	def calculate_y_intercept(self, n, x_sum, y_sum, m):
		b_nominator = y_sum - (m * x_sum)
		b_denominator = n
		b = b_nominator / b_denominator
		
		return b



	def least_squares(self, df):
		n, x_sum, y_sum, xy_sum, x_squared_sum = self.calculate_least_squares_values(df)
		
		#we would like to find the slope for m
		m = self.calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum)
		
		#we would like to find b (y intercept)
		b = self.calculate_y_intercept(n, x_sum, y_sum, m)

		print(f'{m} is slope, {b} is intercept')
		#these are the values for the fitted line
		x_values = df['km']
		y_values = m * x_values + b

		theta0, theta1 = self.gradient_descent(df)

		# Generate a range of mileage values for plotting
		x_range = np.linspace(df['km'].min(), df['km'].max(), 100)

		# Predicted values from least squares
		y_least_squares = m * x_range + b

		# Predicted values from gradient descent
		y_gradient_descent = theta1 * x_range + theta0



	def derivative_MSE(self, mileage, price, intercept, slope):
		n = len(price)
		y = price
		x = mileage
		b = intercept
		c = slope
		h = y - (b + c * x)

		dh_db = -1
		dMSE_db = (2/n) * np.sum(h * dh_db)

		dh_dc = -x
		dMSE_dc = (2/n) * np.sum(h * dh_dc)

		return (dMSE_db, dMSE_dc)
	


	def gradient_descent(self, df):
		mileage = np.array(df['km'])
		price = np.array(df['price'])
		theta0, theta1 = 0, 0
		learning_rate = 0.01
		convergence_threshold = 1e-6
		max_iterations = 6000

		derivative_intercept, derivative_slope = self.derivative_MSE(self, mileage, price, theta0, theta1)
		theta0_prev, theta1_prev = theta0, theta1

		mse_history = []
		mse_history.append(0)
		initial_treshold = 0.0001
		while (max_iterations > 0):
			derivative_intercept, derivative_slope = self.derivative_MSE(mileage, price, theta0, theta1)

			step_size_intercept = learning_rate * derivative_intercept
			step_size_slope = learning_rate * derivative_slope 

			theta0 = theta0_prev - step_size_intercept
			theta1 = theta1_prev - step_size_slope

			mse_current = np.mean((price - (theta0 + theta1 * mileage))**2)
			mse_history.append(mse_current)

			if (max_iterations != 6000 and abs(mse_history[-2] - mse_history[-1]) < initial_treshold):
				print('breaking because MSE didnt change much')
				break 
			initial_treshold *= 0.99
			if (abs(step_size_intercept) < convergence_threshold and abs(step_size_slope) < convergence_threshold):
				print('we reached the convergence treshold')
				break 
			# if (theta0 >= least_squares_intercept and theta1 <= least_squares_slope):
			# 	break 
			
			# print(f'{theta0} compared to {least_squares_intercept} and {theta1} compared to {least_squares_slope}\n')
			theta0_prev = theta0
			theta1_prev = theta1

			max_iterations -= 1
		
		return theta0, theta1


