#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import matplotlib.pyplot as plt
import numpy as np

class PlotRegression:
	def __init__(self, amount_of_iterations_the_training_took, theta0, theta1, mileage, price, mse_history):
		self.amount_of_iterations_the_training_took = amount_of_iterations_the_training_took
		self.mileage = mileage
		self.price = price
		self.theta0 = theta0
		self.theta1 = theta1
		self.mse_history = mse_history


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



	#THIS DOESNT WORK YET, THERE IS SOME BUG SOMEWHERE!
	#ValueError: x and y must have same first dimension, but have shapes (800100,) and (8001,)
	def plot_mse_history(self):
		#take a look how many mse parts do we have?
		plt.figure(figsize=(10, 6))

		n = len(self.mse_history)
		iterations = np.arange(0, n * 100)
		plt.plot(iterations, self.mse_history, 'r-', linewidth=2, label='MSE per 100 Iterations')  # Changed to a red line
		plt.yscale('log')  # Logarithmic scale to show the curve
		plt.title('MSE History')
		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('MSE', fontsize=14)
		plt.legend(loc='upper right', fontsize=12)
		plt.grid(True)
		plt.show()

