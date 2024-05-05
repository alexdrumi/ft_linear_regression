#!/opt/homebrew/bin/python3

import sys
import signal

from linear_regression import LinearRegression


 


	# def plot_linear_regression(self):
	# 	plt.figure(figsize=(10, 6))

	# 	plt.scatter(self.mileage, self.price, color='navy', label='Actual Data', marker='o')

	# 	x_range = np.linspace(self.mileage.min(), self.mileage.max(), 100)

	# 	# Plot least squares regression line
	# 	# plt.plot(x_range, y_least_squares, 'r-', label='Least Squares Regression Line')

	# 	# plot gradient descent regression line
	# 	y_gradient_descent = self.theta1 * x_range + self.theta0

	# 	plt.plot(x_range, y_gradient_descent, 'g--', linewidth=2, label='Gradient Descent Regression Line')
	# 	plt.title('Linear regression with Gradient Descent')
	# 	plt.xlabel('Mileage (km)', fontsize=14)
	# 	plt.ylabel('Price ($)', fontsize=14)
	# 	plt.legend(loc='upper left', fontsize=12)
	# 	plt.grid(True)
	# 	plt.show()


	# def plot_mse_history(self):
	# 	#take a look how many mse parts do we have?
	# 	plt.figure(figsize=(10, 6))

	# 	n = len(self.mse_history)
	# 	iterations = range(0, len(self.mse_history) * 100, 100)
	# 	plt.plot(iterations, self.mse_history, 'r-', linewidth=2, label='MSE per 100 Iterations')  # Changed to a red line
	# 	plt.yscale('log')  # Logarithmic scale to show the curve
	# 	plt.title('MSE History')
	# 	plt.xlabel('Iterations', fontsize=14)
	# 	plt.ylabel('MSE', fontsize=14)
	# 	plt.legend(loc='upper right', fontsize=12)
	# 	plt.grid(True)
	# 	plt.show()

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


def check_data_validity(df):
	# Ensure data has the expected columns and types
	if 'km' not in df or 'price' not in df:
		raise KeyError("Data must include 'km' and 'price' columns.")
	if not pd.api.types.is_numeric_dtype(df['km']) or not pd.api.types.is_numeric_dtype(df['price']):
		raise ValueError("Columns 'km' and 'price' must be numeric.")

def main():
	linear_regression_instance = LinearRegression()
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()

	result = linear_regression_instance.gradient_descent()
	
	linear_regression_instance.plot_linear_regression()
	linear_regression_instance.save_thetas(result)

	linear_regression_instance.plot_mse_history()


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

	linear_regression_instance = LinearRegression()
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()

	result = linear_regression_instance.gradient_descent()
	
	linear_regression_instance.plot_linear_regression()
	linear_regression_instance.save_thetas(result)

	linear_regression_instance.plot_mse_history()

