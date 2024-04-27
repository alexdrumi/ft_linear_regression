#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import signal



'''
#!/opt/homebrew/bin/python3
#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''

class LinearRegression:

	def __init__(self, df):
		self.df = df

		# mileage_normalized = min_max_normalize(df['km'].values)
		# price_normalized = min_max_normalize(df['price'].values)

		# normalized_df = pd.DataFrame({
		# 	'km': mileage_normalized,
		# 	'price': price_normalized
		# })

		#intercept and slope
		self.theta0 = 0
		self.theta1 = 0
		self.theta0_prev = 0
		self.theta1_prev = 0

		#customized parameters based on input flags later
		self.learning_rate = 0.01
		self.convergence_threshold = 1e-6
		self.iterations = 0
		self.max_iterations = 6000
		self.initial_treshold = 0.0001 # decay the treshold over time with 1%

		#derivatives
		self.derivative_intercept = 0
		self.derivative_slope = 0

		#data
		self.mileage = self.min_max_normalize(df['km'].values)
		self.price = self.min_max_normalize(df['price'].values)




	def calculate_least_squares_values(self, df):
		n = df.shape[0]
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



	def derivative_MSE(self):
		y = self.price
		x = self.mileage
		b = self.theta0
		c = self.theta1
		n = len(self.price)
		h = y - (b + c * x)

		dh_db = -1
		dMSE_db = (2/n) * np.sum(h * dh_db)

		dh_dc = -x
		dMSE_dc = (2/n) * np.sum(h * dh_dc)

		return (dMSE_db, dMSE_dc)
	

	def min_max_normalize(self, array):
		x_min = np.min(array)
		x_max = np.max(array)

		array_normalized = (array - x_min)/(x_max - x_min)
		return array_normalized


	def gradient_descent(self, df):
		learning_rate = 0.01
		convergence_threshold = 1e-6
		max_iterations = 6000

		self.derivative_intercept, self.derivative_slope = self.derivative_MSE()
		self.theta0_prev, self.theta1_prev = self.theta0, self.theta1

		mse_history = []
		mse_history.append(0)
		initial_treshold = 0.0001
		while (max_iterations > 0):
			self.derivative_intercept, self.derivative_slope = self.derivative_MSE()

			step_size_intercept = self.learning_rate * self.derivative_intercept
			step_size_slope = self.learning_rate * self.derivative_slope 

			self.theta0 = self.theta0_prev - step_size_intercept
			self.theta1 = self.theta1_prev - step_size_slope

			mse_current = np.mean((self.price - (self.theta0 + self.theta1 * self.mileage))**2)
			mse_history.append(mse_current)

			if (self.max_iterations != 6000 and abs(mse_history[-2] - mse_history[-1]) < self.initial_treshold):
				print('breaking because MSE didnt change much')
				break 
			initial_treshold *= 0.99
			if (abs(step_size_intercept) < convergence_threshold and abs(step_size_slope) < convergence_threshold):
				print('we reached the convergence treshold')
				break 
			# if (theta0 >= least_squares_intercept and theta1 <= least_squares_slope):
			# 	break 
			
			# print(f'{theta0} compared to {least_squares_intercept} and {theta1} compared to {least_squares_slope}\n')
			print(f'{theta0} compared to and {theta1} compared to \n')

			theta0_prev = theta0
			theta1_prev = theta1

			max_iterations -= 1
		
		return theta0, theta1



def signal_handler():
	sys.exit(0)



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)

	df = pd.read_csv('data.csv')
	row, col = df.shape



	linear_regression_instance = LinearRegression(df)

	result = linear_regression_instance.gradient_descent(df)
	print(result)
	# mileage_normalized = min_max_normalize(df['km'].values)
	# price_normalized = min_max_normalize(df['price'].values)

	# normalized_df = pd.DataFrame({
	# 	'km': mileage_normalized,
	# 	'price': price_normalized
	# })

	# theta1, theta2 = self.gradient_descent(normalized_df)

