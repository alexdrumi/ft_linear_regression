#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
#!/opt/homebrew/bin/python3
#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
'''


def plot_straight_line(orig_x, orig_y, x_values, y_values):
	# print(f'{y_values} are y values')
	# y_mean = np.mean(y_values)
	# plt.plot(x_values, y_values, color='blue', label='linear regression')
	# plt.plot([min(orig_x), max(orig_x)], [y_mean, y_mean], color='blue', label='linear regression')

	plt.scatter(orig_x, orig_y, color='red', label='original datapoints')
	plt.plot(x_values, y_values, color='blue', label='linear regression')
	plt.xlabel('km')
	plt.ylabel('price')
	plt.title('Linear regression')
	plt.legend()
	plt.grid(True)
	plt.show()


#calculate values for fitted line
'''

Test values:

n = 7
x_sum = 28
y_sum = 61.8
xy_sum = 314.8
x_squared_sum = 140

'''
def calculate_least_squares_values(df):
	n = df.shape[0]
	xy = df['km'] * df['price']
	x_squared = df['km'] * df['km']

	x_sum = df['km'].sum()
	y_sum = df['price'].sum()
	xy_sum = xy.sum()
	x_squared_sum = x_squared.sum()

	return n, x_sum, y_sum, xy_sum, x_squared_sum


#in the subject, theta1 is m
'''
Test values for x,y to check if all calculates as expected

x: 1,2,3,4,5,6,7
y: 1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16
expected slope m = 2.4142857

#y = mx + b    -> in algebra
#y = b0 + b1x  -> in statistics
#b and b0 is the same
#m and b1 is the same

expected m = 2.4142857142857146
'''
def calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum):
	m_nominator = (n * xy_sum) - (x_sum * y_sum)
	m_denominator = (n * x_squared_sum) - (x_sum ** 2)
	m = m_nominator / m_denominator

	return m


#in the subject, theta0 is m
'''
https://www.youtube.com/watch?v=P8hT5nDai6A&ab_channel=TheOrganicChemistryTutor

Based on the calculate slope test values:
b = -0.828571

'''
def calculate_y_intercept(n, x_sum, y_sum, m):
	b_nominator = y_sum - (m * x_sum)
	b_denominator = n
	b = b_nominator / b_denominator
	
	return b




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
def mean_squared_error(observed_y, predicted_y):
	#this is essentially the SSR divided by the number of observations
	SSR = sum_of_squared_residuals(observed_y, predicted_y)
	number_of_observations = len(df)
	MSE = SSR / number_of_observations

	return MSE


def variation_around_the_mean_of_y(df, y_values):
	mean_of_y = np.mean(df['price'])
	return mean_of_y


#y values could be the mean values but also the y values of the fitted line
def calculate_R2(observed_y, predicted_y_fitted, predicted_y_mean):
	ssr_for_fitted_line = sum_of_squared_residuals(observed_y, predicted_y_fitted)
	ssr_for_mean = sum_of_squared_residuals(observed_y, predicted_y_mean)

	R2 = (ssr_for_mean - ssr_for_fitted_line) / ssr_for_mean
	return R2


#y values could be the mean values but also the y values of the fitted line
def calculate_R2_with_MSE(observed_y, predicted_y_fitted, predicted_y_mean):
	MSE_for_fitted_line = mean_squared_error(observed_y, predicted_y_fitted)
	MSE_for_mean = mean_squared_error(observed_y, predicted_y_mean)

	R2 = (MSE_for_mean - MSE_for_fitted_line) / MSE_for_mean
	return R2




'''
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

'''
def derivative_SSR(mileage, price, intercept, slope):
	y = price
	x = mileage
	b = intercept
	c = slope

	# print(mileage, price, intercept, slope)
	h = y - (b + c * x)

	#derivative of h respect to intercept 
	dh_db = -1

	#derivative of SSR respect to intercept, applying chain rule
	dSSR_db = 2 * h * dh_db

	#derivative of h respect to slope
	dh_dc = -x

	dSSR_dc = 2 * h * dh_dc

	# print(type(dSSR_db), type(dSSR_dc))

	# print(f'{dSSR_db} \n\n')
	''' incorrect values for dSSR_db, not what expected. lets check tomorrow
		>>> a = -2
	>>> b = 3.2
	>>> c = (0+0.5*2.9)
	>>> a * (b - c)
	-3.5000000000000004
	>>> b = 1.9
	>>> c = (0+0.5*2.3)
	>>> a * (b - c)
	-1.5
	>>> b = 1.4
	>>> c = (0+0.5*0.5)
	>>> a * (b - c)
	-2.3
	>>> -3.5000000000000004 + (-1.5) + (-2.3)
	-7.3
	'''

	#sum the derivatives across all data points, return it as a tuple
	sum_derivatives_intercept = np.sum(dSSR_db)
	sum_derivatives_slope = np.sum(dSSR_dc)

	return (sum_derivatives_intercept, sum_derivatives_slope)
	





def gradient_descent(mileage, price):

	# print(mileage, price)
	theta0 = 0 #intercept
	theta1 = 0.5 #slope, influenced by mileage
	learning_rate = 1e-1 #how much do we have to adjust theta
	convergence_treshold = 1e-3 #0.2

	derivative_intercept, derivative_slope = derivative_SSR(mileage, price, theta0, theta1)

	while (abs(derivative_intercept) > convergence_treshold and abs(derivative_slope) > convergence_treshold):
		derivative_intercept, derivative_slope = derivative_SSR(mileage, price, theta0, theta1)

		#update the parameters
		theta0 -= learning_rate * derivative_intercept
		theta1 -= learning_rate * derivative_slope

	return theta0, theta1

	# tmpTheta0 = learning_rate * (1/m) 
	# print(f"{mileage[0]}, {mileage[1]}")
	# test = mileage[0] - (theta0 + thetsa1 * price[0])

	# print(derivative_intercept, derivative_slope)

	# print(f' {mileage}, {price}')
	# print(f'{type(mileage)} is mileage type, {type(price)} is price type')
	# derivative_of_theta0 = derivative_respect_to_theta0(mileage, price, theta0, theta1)
	# print(derivative_of_theta0)




def calculate_R2(observed_y, predicted_y_fitted):
    # Calculate SSR
    ssr = sum_of_squared_residuals(observed_y, predicted_y_fitted)

    # Calculate SST
    mean_y = np.mean(observed_y)
    sst = np.sum((observed_y - mean_y)**2)

    # Calculate R^2
    R2 = 1 - (ssr / sst)
    return R2


'''
https://www.youtube.com/watch?v=P6oIYmK4XdI
When using sum of squared residuals we are using vertical distance instead of perpendicular
'''
#SSR
def sum_of_squared_residuals(observed_y, predicted_y):
	SSR = np.sum((observed_y -  predicted_y)**2)
	return SSR



def least_squares(df):
	n, x_sum, y_sum, xy_sum, x_squared_sum = calculate_least_squares_values(df)
	
	#we would like to find the slope for m
	m = calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum)
	
	#we would like to find b (y intercept)
	b = calculate_y_intercept(n, x_sum, y_sum, m)

	#these are the values for the fitted line
	x_values = df['km']
	y_values = m * x_values + b

	theta0, theta1 = gradient_descent(df['km'], df['price'])
	print(theta0, theta1)

	# #preparing variables for ssr mean
	# mean_for_ssr = np.mean(df['price'])
	# array_mean_for_ssr = np.full(df.shape[0], mean_for_ssr)
	# observed_y = df['price'].values

	# R2 = calculate_R2(observed_y, y_values, array_mean_for_ssr)
	# R2_MSE = calculate_R2_with_MSE(observed_y, y_values, array_mean_for_ssr)

	'''
	R2 tells that approximately 0.7329 (73.29%)
	of the variance in the car prices (independent variable (x)) can be explained by the variance in mileage (dependent variable y)
	relatively high, indicates that the model fits the data well.
	R² is very relevant in regression analysis
	as it provides a measure of how well unseen samples are likely to be predicted by the model, 
	under the assumption that the model assumptions hold true.
	'''

	# print(f'{R2} with simple R2, {R2_MSE} with MSE')


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
	# df = pd.read_csv('datashort.csv')
	df = pd.read_csv('datashort.csv')
	row, col = df.shape

	test1 = np.array([3.2, 1.9, 1.4])
	test2 = np.array([2.9, 2.3, 0.5])

	data_np_row = df['km'].values
	data_np_col = df['price'].values


	ret = gradient_descent(test1, test2)
	#practice dataset just like in the statquest book
	df_test = np.array([2.3, 1.2, 2.7, 1.4, 2.2])
	df_test_mean = df_test.mean()
	# SSR_of_mean = np.sum((df_test - df_test_mean)**2)

	# # print(df_test.mean())
	# # print(f'{test_mean} is testmean')
	# # print(f'{df_test["test_values"]} is testvalues')

	# data_organic_chem_tutor = {
	# 	'km': [1,2,3,4,5,6,7],
	# 	'price': [1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16.0]}


	# data_statquest = {
	# 	'km': [1,2,3,4,5],
	# 	'price': [2.3, 1.2, 2.7, 1.4, 2.2]
	# }

	# test data from statquest book	
	# observed_y = np.array([1.2, 2.2, 1.4, 2.7, 2.3])
	# predicted_y = np.array([1.1, 1.8, 1.9, 2.4, 2.5])
	# SSR_np = np.sum((observed_y - predicted_y)**2)
	# print(SSR_np)

	# df_organic_chem_tutor = pd.DataFrame(data_organic_chem_tutor)
	# df_statquest = pd.DataFrame(data_statquest)

	# least_squares(df)

