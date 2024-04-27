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
		
		self.derivative_intercept, self.derivative_slope = self.derivative_MSE()
		self.theta0_prev, self.theta1_prev = self.theta0, self.theta1

		mse_history = []
		mse_history.append(0)
		while (self.max_iterations > 0):
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
			self.initial_treshold *= 0.99
			if (abs(step_size_intercept) < self.convergence_threshold and abs(step_size_slope) < self.convergence_threshold):
				print('we reached the convergence treshold')
				break 
		
			self.theta0_prev = self.theta0
			self.theta1_prev = self.theta1

			self.max_iterations -= 1
		
		return self.theta0, self.theta1



	def plot_linear_regression(self):
		plt.figure(figsize=(10, 6))

		# plot actual data points
		plt.scatter(self.mileage, self.price, color='blue', label='Actual Data')

		x_range = np.linspace(self.mileage.min(), self.mileage.max(), 100)

		# Plot least squares regression line
		# plt.plot(x_range, y_least_squares, 'r-', label='Least Squares Regression Line')

		# plot gradient descent regression line
		y_gradient_descent = self.theta1 * x_range + self.theta0

		plt.plot(x_range, y_gradient_descent, 'g--', label='Gradient Descent Regression Line')
		plt.title('Linear regression with Gradient Descent')
		plt.xlabel('Mileage (km)')
		plt.ylabel('Price ($)')
		plt.legend()
		plt.grid(True)
		plt.show()


	


def signal_handler():
	sys.exit(0)



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)

	df = pd.read_csv('data.csv')
	row, col = df.shape


	linear_regression_instance = LinearRegression(df)

	result = linear_regression_instance.gradient_descent(df)
	linear_regression_instance.plot_linear_regression()

