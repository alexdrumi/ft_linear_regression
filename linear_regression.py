#!/opt/homebrew/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
'''
#!/opt/homebrew/bin/python3
#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''


class LinearRegression:

	def __init__(self, learning_rate, convergence_treshold, max_iterations=8000):
		self.df = None
		self.filename = 'training_data.csv'

		self.theta0 = 0 #intercept
		self.theta1 = 0 #slope
		
		self.learning_rate = learning_rate
		self.convergence_threshold = convergence_treshold
		self.max_iterations = max_iterations
		self.iterations = 0
		# self.initial_treshold = 0.0001 originally for dynamic decay

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



	def assign_mileage_and_price(self):
			self.mileage = self.min_max_normalize(self.df['km'].values)
			self.price = self.min_max_normalize(self.df['price'].values)



	def read_csv(self):
		try:
			df = pd.read_csv(self.filename)
			self.df = df
		except (FileNotFoundError, PermissionError, IOError) as e:
			self.handle_file_error(e)



	def handle_file_error(self, error):

		if isinstance(error, FileNotFoundError):
			print(f'File not found: {self.filename}')
		elif isinstance(error, PermissionError):
			print(f'Permission denied while trying to open the file: {self.filename}')
		else:
			print(f'I/O error occured while reading the file: {self.filename}')
			print(f'Error details: {error}')



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
	# 	dh_db = -1
	# 	# derivative of SSR respect to intercept, applying chain rule (this is 1/m in the subject)
	# 	dMSE_db = (2/n) * np.sum(h * dh_db)
  
	# 	# derivative of h respect to slope
	# 	dh_dc = -x
	# 	# derivative of SSR respect to slope, applying chain rule, divided by n (this is 1/m in the subject)
	# 	dMSE_dc = (2/n) * np.sum(h * dh_dc)
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
		print(f"{abs(self.step_size_intercept)} is step_size_intercept\n{abs(self.step_size_slope)} is step_size_slope\n")
		intercept_converged = abs(self.step_size_intercept) < self.convergence_threshold
		slope_converged = abs(self.step_size_slope) < self.convergence_threshold
		
		result = intercept_converged and slope_converged
		# print(f'{result} is from the convergence')
		return result



	def log_MSE(self):
		mse_current = self.compute_MSE()
		self.mse_history.append(mse_current)



	def save_thetas(self, thetas):
		with open('thetas.txt', 'w') as file:
			print(f'{thetas[0]}, {thetas[1]}')
			file.write(str(thetas[0]) + "\n")
			file.write(str(thetas[1]) + "\n")
		file.close()



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

			if self.convergence_succeeded() == True:
				print('True from the loop convergence\n')
				break

			self.iterations += 1
		
		return self.theta0, self.theta1



	def plot_linear_regression(self):
		plt.figure(figsize=(10, 6))

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



	def print_to_terminal(self):
		print(f"\033[92mTraining Update:"
			f"\n\033[92mIterations it took to train: \033[97m{self.iterations}"
			f"\n\033[92mLearning Rate: \033[97m{self.learning_rate}"
			f"\n\033[92mMax Iterations: \033[97m{self.max_iterations}"
			f"\n\033[92mConvergence Threshold: \033[97m{self.convergence_threshold}"
			f"\n\033[92mTheta0 (Intercept): \033[97m{self.theta0}"
			f"\n\033[92mTheta1 (Slope): \033[97m{self.theta1}"
			f"\n\033[92mMSE: \033[97m{self.compute_MSE()}"
			f"\n\033[92mRMSE: \033[97m{self.compute_RMSE()}")
