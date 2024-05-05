#!/opt/homebrew/bin/python3

#read thetas from file
#predict a given value from the terminal

import numpy as np
import pandas as pd
'''
!/opt/homebrew/bin/python3
!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''

'''
Normalizes a value using min-max normalization
'''
def min_max_normalize(value, min_val, max_val):
	return (value - min_val) / (max_val - min_val)


'''
Loads the trained theta values and uses them to predict price based on a given mileage.
'''
def load_thetas():
	with open('thetas.txt', 'r') as file:
		theta0 = float(file.readline().strip())
		theta1 = float(file.readline().strip())
	return theta0, theta1

'''
Predicts the price based on mileage using the trained thetas and min/max values for normalization
'''
def predict_price(mileage, theta0, theta1, min_km, max_km, min_price, max_price):
	normalized_mileage = min_max_normalize(mileage, min_km, max_km)
	# Predict price in normalized scale
	normalized_price = theta1 * normalized_mileage + theta0
	# Convert normalized price back to dollar price
	price_in_dollars = normalized_price * (max_price - min_price) + min_price
	return price_in_dollars




if __name__ == '__main__':
	# Load thetas
	# read mileage arguiments from the command line

	#error check?
	mileage = float(input("Enter the mileage of the car: "))
	df = pd.read_csv('data.csv')

	min_km, max_km = np.min(df['km'].values), np.max(df['km'].values)
	min_price, max_price = np.min(df['price'].values), np.max(df['price'].values)
	theta0, theta1 = load_thetas()

	# These min and max values should be the same as used during training normalization
	# min_km = 5000  # Example min km, replace with actual min value from tra
	prediction = predict_price(mileage, theta0, theta1, min_km, max_km, min_price, max_price)
	print(f'The estimated price is: {prediction}$')



