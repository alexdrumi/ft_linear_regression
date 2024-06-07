import pandas as pd
from linear_regression import LinearRegression
from error_handler import ErrorHandler

def main():
	error_handler = ErrorHandler()
	learning_rate = 0.01
	convergence_threshold = 0.0000001
	linear_regression_instance = LinearRegression(learning_rate, convergence_threshold)
	theta0, theta1 = linear_regression_instance.load_thetas()

	try:
		df = pd.read_csv('../data/data.csv')
	except Exception as e:
		error_handler.handle_error(e, '../data/data.csv')
		return

	try:
		km_min = df['km'].min()
		km_max = df['km'].max()
		price_min = df['price'].min()
		price_max = df['price'].max()
	except KeyError as e:
		error_handler.handle_error(e, '../data/data.csv')
		return

	try:
		mileage = float(input("\033[33m\nEnter the mileage of the car: \033[0m"))
		if mileage < 0 or mileage > km_max:
			error_handler.log_message(f"Invalid input. Please enter a mileage between 0 and {km_max}.")
			return
	except ValueError:
		error_handler.log_message("Invalid input. Please enter a valid number for mileage.")
		return

	
	try:
		predicted_price = linear_regression_instance.predict_price(
			mileage,
			theta0,
			theta1,
			km_min,
			km_max,
			price_min,
			price_max
		)
		
		print(f"The estimated price for a car with {mileage} km is: ${predicted_price:.2f}")
	except Exception as e:
		error_handler.handle_error(e)

if __name__ == '__main__':
	main()
