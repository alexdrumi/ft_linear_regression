#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3

#read thetas from file
#predict a given value from the terminal
'''
!/opt/homebrew/bin/python3
!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''

import pandas as pd
from linear_regression import LinearRegression

def main():
	learning_rate = 0.01
	convergence_threshold = 0.0000001
	linear_regression_instance = LinearRegression(learning_rate, convergence_threshold)
	linear_regression_instance.load_thetas()

	#read mileage from user
	try:
		mileage = float(input("Enter the mileage of the car: "))
	except ValueError:
		print("Invalid input. Please enter a valid number for mileage.")
		return

	# Assuming min and max values are stored or can be loaded
	try:
		df = pd.read_csv('data.csv')
	except FileNotFoundError:
		print("Data file not found. Please ensure the data.csv is available.")
		return
	except pd.errors.EmptyDataError:
		print("Data file is empty.")
		return

	# Predict the price
	predicted_price = linear_regression_instance.predict_price(linear_regression_instance.mileage,
		linear_regression_instance.theta0,
		linear_regression_instance.theta1,
		df['km'].min(), 
		df['km'].max(), 
		df['price'].min(), 
		df['price'].max()
	)
	print(f"The estimated price for a car with {mileage} km is: ${predicted_price:.2f}")



if __name__ == '__main__':
	main()
