#!/opt/homebrew/bin/python3
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

	def __init__(self):
		self.df = None
		self.filename = 'data.csv'

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
		self.step_size_intercept = 0
		self.step_size_slope = 0

		#derivatives
		self.derivative_intercept = 0
		self.derivative_slope = 0

		#data
		self.mileage = 0 #self.min_max_normalize(self.df['km'].values)
		self.price = 0 #self.min_max_normalize(self.df['price'].values)

		#mse history
		self.mse_history = []
		self.mse_history.append(0)



	def compute_MSE(self):
		prediction = self.theta1 * self.mileage + self.theta0
		MSE = np.mean((self.price - (prediction) **2))
		return MSE


	#b ->intercept
	#c ->slope
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
	


	def gradient_descent(self):
		
		self.derivative_intercept, self.derivative_slope = self.derivative_MSE()
		self.theta0_prev, self.theta1_prev = self.theta0, self.theta1

		
		while (self.max_iterations > 0):
			self.derivative_intercept, self.derivative_slope = self.derivative_MSE()

			self.step_size_intercept = self.learning_rate * self.derivative_intercept
			self.step_size_slope = self.learning_rate * self.derivative_slope 

			self.theta0 -= self.step_size_intercept
			self.theta1 -= self.step_size_slope

			if (self.max_iterations % 100 == 0):
				mse_current = self.compute_MSE()
				self.mse_history.append(mse_current)
			#we can use this for logging and monitoring how MSE behaves
	
			if (self.convergence_succeeded() is True):
				print('We have reached the convergence treshold.')
				break 
		
			self.theta0_prev = self.theta0
			self.theta1_prev = self.theta1

			self.max_iterations -= 1
		
		return self.theta0, self.theta1




	def min_max_normalize(self, array):
		x_min = np.min(array)
		x_max = np.max(array)

		array_normalized = (array - x_min)/(x_max - x_min)
		return array_normalized



	def	convergence_succeeded(self):
		return (abs(self.step_size_intercept) < self.convergence_threshold 
				and abs(self.step_size_slope) < self.convergence_threshold)



	def plot_linear_regression(self):
		plt.figure(figsize=(10, 6))

		# plot actual data points
		plt.scatter(self.mileage, self.price, color='navy', label='Actual Data', marker='o')

		x_range = np.linspace(self.mileage.min(), self.mileage.max(), 100)

		# Plot least squares regression line
		# plt.plot(x_range, y_least_squares, 'r-', label='Least Squares Regression Line')

		# plot gradient descent regression line
		y_gradient_descent = self.theta1 * x_range + self.theta0

		plt.plot(x_range, y_gradient_descent, 'g--', linewidth=2, label='Gradient Descent Regression Line')
		plt.title('Linear regression with Gradient Descent')
		plt.xlabel('Mileage (km)', fontsize=14)
		plt.ylabel('Price ($)', fontsize=14)
		plt.legend(loc='upper left', fontsize=12)
		plt.grid(True)
		plt.show()



	def plot_mse_history(self):
		#take a look how many mse parts do we have?
		plt.figure(figsize=(10, 6))

		n = len(self.mse_history)
		iterations = range(0, len(self.mse_history) * 100, 100)
		plt.plot(iterations, self.mse_history, 'r-', linewidth=2, label='MSE per 100 Iterations')  # Changed to a red line
		plt.yscale('log')  # Logarithmic scale to show the curve
		plt.title('MSE History')
		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('MSE', fontsize=14)
		plt.legend(loc='upper right', fontsize=12)
		plt.grid(True)
		plt.show()




	def read_csv(self):
		try:
			df = pd.read_csv(self.filename)
			self.df = df
		except (FileNotFoundError, PermissionError, IOError) as e:
			self.handle_file_error(e)



	def assign_mileage_and_price(self):
		self.mileage = self.min_max_normalize(self.df['km'].values)
		self.price = self.min_max_normalize(self.df['price'].values)
		return 


	def handle_file_error(self, error):

		if isinstance(error, FileNotFoundError):
			print(f'File not found: {self.filename}')
		elif isinstance(error, PermissionError):
			print(f'Permission denied while trying to open the file: {self.filename}')
		else:
			print(f'I/O error occured while reading the file: {self.filename}')
			print(f'Error details: {error}')



	def save_thetas(self, thetas):
		with open('thetas.txt', 'w') as file:
			print(f'{thetas[0]}, {thetas[1]}')
			file.write(str(thetas[0]) + "\n")
			file.write(str(thetas[1]) + "\n")
		file.close()



def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} ({signum}), exciting program.')
	sys.exit(0)



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)

	# df = pd.read_csv('data.csv')
	# row, col = df.shape

	linear_regression_instance = LinearRegression()
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()

	result = linear_regression_instance.gradient_descent()
	
	linear_regression_instance.plot_linear_regression()
	linear_regression_instance.save_thetas(result)

	linear_regression_instance.plot_mse_history()
