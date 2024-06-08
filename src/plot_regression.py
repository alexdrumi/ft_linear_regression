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
		y_gradient_descent = self.theta1 * x_range + self.theta0
		plt.plot(x_range, y_gradient_descent, 'g--', linewidth=2, label='Gradient Descent Regression Line')
		plt.title('Linear regression with Gradient Descent')
		plt.xlabel('Mileage (km)', fontsize=14)
		plt.ylabel('Price ($)', fontsize=14)
		plt.legend(loc='upper left', fontsize=12)
		plt.grid(True)
		plt.show()



	def plot_mse_history(self):
		iterations = np.arange(len(self.mse_history))
		plt.figure(figsize=(10, 6))
		plt.plot(iterations, self.mse_history, 'r-', linewidth=2, label='MSE per 100 Iterations')  # Changed to a red line		plt.yscale('log')  # Logarithmic scale for y-axis
		plt.title('MSE History')
		plt.xlabel('Iterations', fontsize=14)
		plt.ylabel('MSE', fontsize=14)
		plt.legend(loc='upper right', fontsize=12)
		plt.grid(True)
		plt.show()



