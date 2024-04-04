#!/opt/homebrew/bin/python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
#y = mx + b    -> in algebra
#y = b0 + b1x  -> in statistics
#b and b0 is the same
#m and b1 is the same

test values:

n = 7
x_sum = 28
y_sum = 61.8
xy_sum = 314.8
x_squared_sum = 140

expected m = 2.4142857142857146
'''

def plot_straight_line(orig_x, orig_y, x_values, y_values):
	plt.scatter(orig_x, orig_y, color='red', label='original datapoints')
	plt.scatter(x_values, y_values, color='blue', label='linear regression')
	plt.xlabel('km')
	plt.ylabel('price')
	plt.title('Linear regression')
	plt.legend()
	plt.grid(True)
	plt.show()



def calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum):
	m_nominator = (n * xy_sum) - (x_sum * y_sum)
	m_denominator = (n * x_squared_sum) - (x_sum ** 2)
	m = m_nominator / m_denominator

	return m



def calculate_b_intercept(n, x_sum, y_sum, m):
	b_nominator = y_sum - (m * x_sum)
	b_denominator = n
	b = b_nominator / b_denominator

	return b



def calculate_least_squares_values(df):
	n = df.shape[0]
	xy = df['km'] * df['price']
	x_squared = df['km'] * df['km']

	x_sum = df['km'].sum()
	y_sum = df['price'].sum()
	xy_sum = xy.sum()
	x_squared_sum = x_squared.sum()

	return n, x_sum, y_sum, xy_sum, x_squared_sum

'''
We could have divided every single (Observedi - Predictedi)ˆ2 individually like: 
(Observed1 - Predicted1)ˆ2 / 2 
(Observed2 - Predicted2)ˆ2 / 2 

It would give us an a result but computationally expensive since its uses division, eg:
observed1 = 10
predicted1 - 5

observed2 = 12
predicted2 = 6

Wrong method: (10-5)ˆ2 = 25/2 , (12-6)ˆ2 = 36/2 
Total : 12.5 + 18 = 30.5

Correct method: (10-5)ˆ2 = 25 , (12-6)ˆ2 = 36 
Total : 25 + 36 = 61 / 2

'''
def mean_squared_error(df, y_values):
	#this is essentially the SSR divided by the number of observations
	SSR = sum_of_squared_residuals(df, y_values)
	number_of_observations = len(df)
	MSE = SSR / number_of_observations

	return MSE

'''
https://www.youtube.com/watch?v=P6oIYmK4XdI
When using sum of squared residuals we are using vertical distance instead of perpendicular
'''
def sum_of_squared_residuals(df, y_values):
	SSR = np.sum((df['price'] -  y_values)**2)
	# SSR = 0
	# for item0, item1, item2 in zip(df['km'], df['price'], y_values):
	# 	# print(f'At km: {item0}: {item1} is orig, {item2} is the line')
	# 	observed_min_predicted_value_squared = (item1 - item2)**2
	# 	SSR += observed_min_predicted_value_squared
	# 	print(f'The sum of squared residuals is : {SSR}')
		#these are the actual differences for SSE
	# this seems waaay to big. maybe we should normalize stuff?
	return SSR


def least_squares(df):
	n, x_sum, y_sum, xy_sum, x_squared_sum = calculate_least_squares_values(df)
	
	#we would like to find the slope for m
	m = calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum)
	
	#we would like to find b (y intercept) based on knowing m
	b = calculate_b_intercept(n, x_sum, y_sum, m)

	x_values = df['km']
	y_values = m * x_values + b

	# frames = [df['price'], y_values]
	df3 = pd.DataFrame({
		"Original Price" : [df['price']],
		"Straight line " : [y_values],
	})

	#what is y value of the line at coordinate x
	#what is y value of the original datapoint at coordinate x

	# print(type(df['km']), type(y_values))
	# plot_straight_line(df['km'].values, df['price'].values, x_values, y_values)
	# sum_of_squared_residuals(df, y_values)
	MSE = mean_squared_error(df, y_values)
	print(MSE)



def create_graph_for_three(df):
	data_np_row_three = df['km'].head(3).values
	data_np_col_three = df['price'].head(3).values
	
	plt.plot(data_np_row_three, data_np_col_three, label='price estimation', marker='o', linestyle='')
	plt.grid(True)
	plt.show()



if __name__ == "__main__":

	#maybe include guard in case of failure
	df = pd.read_csv('datashort.csv')
	row, col = df.shape

	data_np_row = df['km'].values
	data_np_col = df['price'].values

	# create_graph_for_three(df)
	least_squares(df)
