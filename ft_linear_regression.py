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
	plt.plot(x_values, y_values, color='blue', label='linear regression')
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


#fitted line
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


def variation_around_the_mean_of_y(df, y_values):
	mean_of_y = np.mean(df['price'])
	return mean_of_y



def sum_of_squared_residuals_for_mean_np(df):
	numpy_dataset = np.array(df)
	mean = numpy_dataset.mean()
	SSR_np = np.sum((numpy_dataset - mean)**2)

	return SSR_np



def sum_of_squared_residuals_for_mean_df(df, mean):
	SSR_for_mean = np.sum((df['price'] - mean)**2)
	return SSR_for_mean



def calculate_R2(df, y_values):
	# var_mean = variation_around_the_mean_of_y(df, y_values)
	# ssr_for_mean = sum_of_squared_residuals_for_mean_df(df, var_mean)
	ssr_for_mean = sum_of_squared_residuals_for_mean_np(df)
	print(ssr_for_mean)



'''
https://www.youtube.com/watch?v=P6oIYmK4XdI
When using sum of squared residuals we are using vertical distance instead of perpendicular
'''
def sum_of_squared_residuals(df, y_values):
	SSR = np.sum((df['price'] -  y_values)**2)
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
	calculate_R2(df, y_values)

	#what is y value of the line at coordinate x
	#what is y value of the original datapoint at coordinate x

	# print(type(df['km']), type(y_values))
	plot_straight_line(df['km'].values, df['price'].values, x_values, y_values)
	# sum_of_squared_residuals(df, y_values)
	# MSE = mean_squared_error(df, y_values)
	# print(MSE)
	# plot_straight_line(df['km'], df['price'], x_values, y_values)



def create_graph_for_three(df):
	data_np_row_three = df['km'].head(3).values
	data_np_col_three = df['price'].head(3).values
	
	plt.plot(data_np_row_three, data_np_col_three, label='price estimation', marker='o', linestyle='')
	plt.grid(True)
	plt.show()

#we will eventually have to convert the pd.read_csv to np array
# df = pd.read_csv('your_file.csv')

# # Convert DataFrame to NumPy array
# data = df.to_numpy()
if __name__ == "__main__":

	#maybe include guard in case of failure
	df = pd.read_csv('datashort.csv')
	row, col = df.shape

	data_np_row = df['km'].values
	data_np_col = df['price'].values

	#practice dataset just like in the statquest book
	df_test = np.array([2.3, 1.2, 2.7, 1.4, 2.2])
	df_test_mean = df_test.mean()
	# SSR_of_mean = np.sum((df_test - df_test_mean)**2)

	# # print(df_test.mean())
	# # print(f'{test_mean} is testmean')
	# # print(f'{df_test["test_values"]} is testvalues')


	# print(SSR_of_mean)
	# create_graph_for_three(df)
	least_squares(df)
	# SSR_for_mean = sum_of_squared_residuals_for_mean_np(df_test)
	# print(SSR_for_mean)
