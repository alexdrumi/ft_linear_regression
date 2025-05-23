import pandas as pd
import numpy as np
import os

from error_handler import ErrorHandler

class LinearRegression:

	def __init__(self, learning_rate, convergence_treshold, max_iterations=8000):
		self.df = None
		script_dir = os.path.dirname(__file__)
		self.filename = os.path.join(script_dir, '../data/training_data.csv')
		self.theta_file = os.path.join(script_dir, '../data/thetas.txt')

		self.theta0 = 0 #intercept
		self.theta1 = 0 #slope

		self.learning_rate = learning_rate
		self.convergence_threshold = convergence_treshold
		self.max_iterations = max_iterations
		self.iterations = 0

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

		#error handler
		self.error_handler = ErrorHandler()



	def assign_mileage_and_price(self):
		km_min = self.df['km'].min()
		km_max = self.df['km'].max()
		price_min = self.df['price'].min()
		price_max = self.df['price'].max()

		if km_min == km_max:
			raise ValueError("Zero variance in 'km' column.")
		if price_min == price_max:
			raise ValueError("Zero variance in 'price' column.")

		self.mileage = self.min_max_normalize(self.df['km'].values, km_min, km_max)
		self.price = self.min_max_normalize(self.df['price'].values, price_min, price_max)



	def read_csv(self):
		df = pd.read_csv(self.filename)
		self.df = df



	def compute_MSE(self):
		predictions = self.theta0 + (self.theta1 * self.mileage) #theta0 = intercept + slope * mileage
		return np.mean((self.price - predictions) ** 2)



	def compute_RMSE(self):
		return np.sqrt(self.compute_MSE())



	def compute_MAE(self):
		predictions = self.theta0 + self.theta1 * self.mileage
		mae = np.mean(np.abs(self.price - predictions))
		return mae



	'''
	------------------------------------------------------------------------------
	The following is calculating the partial derivatives respect to slope and intercept
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



	'''
	residuals = price - estimatePrice(mileage) -> price - (theta0 + theta1 * mileage) 
	this gives us the differences between actual prices and estimated prices.

	substituting in the provided formulas:
	tmp0 = learningRate * (1/m) * sum(residuals)
	tmp1 = learningRate * (1/m) * sum(residuals * mileage)
	
	in the code below, dMSE_db and dMSE_dc are the gradients of the MSE with respect to 
	theta0 (intercept) and theta1 (slope) respectively. The learning rate is applied in 
	the main gradient_descent function.
	'''
	def compute_MSE_gradients(self):
		x, y = self.mileage, self.price
		b, c = self.theta0, self.theta1
		n = len(self.price)

		residuals = y - (b + c * x)
		
		dMSE_db = -2 * np.mean(residuals) #gradient with respect to theta0 (intercept) this is a faster calculation than summing and dividing/multiplying
		dMSE_dc = -2 * np.mean(residuals * x) #gradient with respect to theta1 (slope) this is a faster calculation than summing and dividing/multiplying

		return dMSE_db, dMSE_dc



	def min_max_normalize(self, value, min_value, max_value):
		if min_value == max_value:
			return np.zeros_like(value) if isinstance(value, np.ndarray) else 0
		return (value - min_value) / (max_value - min_value)



	def convergence_succeeded(self):
		intercept_converged = abs(self.step_size_intercept) < self.convergence_threshold
		slope_converged = abs(self.step_size_slope) < self.convergence_threshold
		result = intercept_converged and slope_converged
		return result



	def log_MSE(self):
		mse_current = self.compute_MSE()
		self.mse_history.append(mse_current)



	def save_thetas(self, thetas):
		with open('thetas.txt', 'w') as file:
			file.write(str(thetas[0]) + "\n")
			file.write(str(thetas[1]) + "\n")
		file.close()



	def load_thetas(self):
		with open('thetas.txt', 'r') as file:
			theta0 = float(file.readline().strip())
			theta1 = float(file.readline().strip())
		return theta0, theta1



	def estimate_price(self, mileage, theta0, theta1, min_km, max_km, min_price, max_price):
		normalized_mileage = self.min_max_normalize(mileage, min_km, max_km)
		normalized_price = theta0 + (theta1 * normalized_mileage)
		predicted_price = normalized_price * (max_price - min_price) + min_price
		return predicted_price



	'''
	Take a look at RMSprop, Adam, or incorporating learning rate schedules eventually
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
				break

			self.iterations += 1

		return self.theta0, self.theta1



	def print_to_terminal(self):
		print(f"\n\033[92mTraining Update:"
			f"\n\033[92mIterations it took to train: \033[97m{self.iterations}"
			f"\n\033[92mLearning Rate: \033[97m{self.learning_rate}"
			f"\n\033[92mMax Iterations: \033[97m{self.max_iterations}"
			f"\n\033[92mConvergence Threshold: \033[97m{self.convergence_threshold}"
			f"\n\033[92mTheta0 (Intercept): \033[97m{self.theta0}"
			f"\n\033[92mTheta1 (Slope): \033[97m{self.theta1}"
			f"\n\033[92mMSE: \033[97m{self.compute_MSE()}"
			f"\n\033[92mRMSE: \033[97m{self.compute_RMSE()}\n\n",
			f"\n\033[92mLINEAR REGRESSION TRAINING IS COMPLETE.\033[97m")



	def run_linear_regression(self):
		try:
			self.read_csv()
			self.assign_mileage_and_price()
			result = self.gradient_descent()
			return result
		except (FileNotFoundError, PermissionError, IOError, ValueError) as e:
			self.error_handler.handle_error(e, self.filename)
