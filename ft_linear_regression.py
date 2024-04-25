#!/opt/homebrew/bin/python3

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
	# n = 3
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

divide the whole thing with the amount of datapoints to get the MSE
'''
def derivative_MSE(mileage, price, intercept, slope):
	y = price
	x = mileage
	b = intercept
	c = slope
	n = len(price)
	# print(f'{price[0]} = price0, {b} = intercept, {mileage[0]} = mileage')
	# print(-2 * (price[0] - (0 + b * mileage[0])))

	h = y - (b + c * x)

	#derivative of h respect to intercept 
	dh_db = -1

	#derivative of SSR respect to intercept, applying chain rule
	dMSE_db = (2/n) * np.sum(h * dh_db)

	#derivative of h respect to slope
	dh_dc = -x

	dMSE_dc = (2/n) * np.sum(h * dh_dc)

	# print(type(dSSR_db), type(dSSR_dc))

	# print(f'{dSSR_db} \n\n')
	# ''' incorrect values for dSSR_db, not what expected. lets check tomorrow
	# 	>>> a = -2
	# >>> b = 3.2
	# >>> c = (0+0.5*2.9)
	# >>> a * (b - c)
	# -3.5000000000000004
	# >>> b = 1.9
	# >>> c = (0+0.5*2.3)
	# >>> a * (b - c)
	# -1.5
	# >>> b = 1.4
	# >>> c = (0+0.5*0.5)
	# >>> a * (b - c)
	# -2.3
	# >>> -3.5000000000000004 + (-1.5) + (-2.3)
	# -7.3
	# '''

	#sum the derivatives across all data points, return it as a tuple
	# sum_derivatives_intercept = np.sum(dMSE_db)
	# sum_derivatives_slope = np.sum(dMSE_dc)

	return (dMSE_db, dMSE_dc)
	

#to avoid local minima early in the training
def random_theta_initialization():
	return np.random.rand(), np.random.rand()


'''

Convergence Check:
The given examples use a norm or absolute difference check between the cost function evaluations from one iteration to the next. 
For a linear regression, this would typically involve computing the MSE or another loss function,
but the current setup seems to involve checking the parameter changes directly, could also work.
The values are a bit tricky at the moment, they change too much. Not sure just yet why.

'''
# def gradient_descent(mileage, pric:e):
def gradient_descent(df):
	#0.9392969779416096 -1.0035178010925934 should be end result
	mileage = np.array(df['km'])
	price = np.array(df['price'])

	# theta0, theta1 = random_theta_initialization() #intercept and slope respectively
	# theta1 = 0.5 
	theta0, theta1 = 0, 0
	learning_rate = 0.01 #with smaller learning rate, the adjustments are also smaller thus the max_iteration has to be raised
	#if you change this for instance 0.1, it will barely iterate but will not be precise
	convergence_threshold = 1e-6
	max_iterations = 6000
	#with smaller learning rate this becomes a bit more precise but max_iterations have to be raised
	least_squares_intercept = 0.9393189294497466
	least_squares_slope = -1.003575742397017

	derivative_intercept, derivative_slope = derivative_MSE(mileage, price, theta0, theta1)
	theta0_prev, theta1_prev = theta0, theta1

	# theta1_values = np.linspace(-0.1, 0.1, 100)
	# theta0_values = np.linspace(-2000, 2000, 100)
	# Theta1, Theta0 = np.meshgrid(theta1_values, theta0_values)

	# # Compute MSE for each combination of theta0 and theta1
	# MSE = np.array([np.mean((theta0 + theta1 * df['km'] - df['price'])**2) for theta0, theta1 in zip(np.ravel(Theta0), np.ravel(Theta1))])
	# MSE = MSE.reshape(Theta0.shape)

	# # Plot the MSE surface
	# plt.figure(figsize=(10, 8))
	# cp = plt.contourf(Theta1, Theta0, MSE, levels=np.logspace(0, 5, 35), cmap='viridis')
	# plt.colorbar(cp)
	# plt.title('MSE Loss Surface')
	# plt.xlabel('Theta1 (slope)')
	# plt.ylabel('Theta0 (intercept)')
 
	# Fast Convergence: If the MSE drops quickly and flattens out, it suggests that the learning rate is effectively tuned.
	# Slow Convergence or Non-Convergence: If the MSE decreases very slowly or oscillates, consider adjusting the learning rate or increasing the iteration count.


	mse_history = []
	mse_history.append(0)
	initial_treshold = 0.0001
	while (max_iterations > 0):
		derivative_intercept, derivative_slope = derivative_MSE(mileage, price, theta0, theta1)

		step_size_intercept = learning_rate * derivative_intercept
		step_size_slope = learning_rate * derivative_slope 

		# print(derivative_intercept, learning_rate_intercept)
		theta0 = theta0_prev - step_size_intercept #gradient * stepsize
		theta1 = theta1_prev - step_size_slope #gradient * stepsize

		#mse to monitor
		mse_current = np.mean((price - (theta0 + theta1 * mileage))**2)
		mse_history.append(mse_current)

		# print(f'{mse_history[-2]} previous, {mse_history[-1]} current\n')
		# Check for convergence based on parameter changes, this might be interesting
		# if np.linalg.norm([theta0 - theta0_prev, theta1 - theta1_prev]) < convergence_threshold:
		# 	break
		#if both thetas succesfully achieved convergence, we stop iterating
		if (max_iterations != 6000 and abs(mse_history[-2] - mse_history[-1]) < initial_treshold):
			print('breaking because MSE didnt change much')
			break 
		# decay the treshold over time with 1%
		initial_treshold *= 0.99
		if (abs(step_size_intercept) < convergence_threshold and abs(step_size_slope) < convergence_threshold):
			print('we reached the convergence treshold')
			break 
		if (theta0 >= least_squares_intercept and theta1 <= least_squares_slope):
			break 
		
		print(f'{theta0} compared to {least_squares_intercept} and {theta1} compared to {least_squares_slope}\n')
		#save prev values
		theta0_prev = theta0
		theta1_prev = theta1

		print(theta0_prev, theta1_prev, max_iterations)
		max_iterations -= 1
	# 	plt.scatter(theta1, theta0, color='red')  # Plot current theta values

	# plt.show()

	return theta0, theta1



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

	print(f'{m} is slope, {b} is intercept')
	#these are the values for the fitted line
	x_values = df['km']
	y_values = m * x_values + b

	# # Plotting the original data points
	# plt.scatter(df['km'], df['price'], color='blue', label='Data Points')

	# # Plotting the regression line
	# plt.plot(x_values, y_values, color='red', label='Fitted Line')

	# # Adding labels and legend
	# plt.xlabel('Kilometers')
	# plt.ylabel('Price')
	# plt.title('Linear Regression Fit')
	# plt.legend()

	# # Display the plot
	# plt.show()
	theta0, theta1 = gradient_descent(df)



	# Generate a range of mileage values for plotting
	x_range = np.linspace(df['km'].min(), df['km'].max(), 100)

	# Predicted values from least squares
	y_least_squares = m * x_range + b

	# Predicted values from gradient descent
	y_gradient_descent = theta1 * x_range + theta0

	# plt.figure(figsize=(10, 6))

	# # Plot actual data points
	# plt.scatter(df['km'], df['price'], color='blue', label='Actual Data')

	# # Plot least squares regression line
	# plt.plot(x_range, y_least_squares, 'r-', label='Least Squares Regression Line')

	# # Plot gradient descent regression line
	# plt.plot(x_range, y_gradient_descent, 'g--', label='Gradient Descent Regression Line')

	# plt.title('Comparison of Regression Methods')
	# plt.xlabel('Mileage (km)')
	# plt.ylabel('Price ($)')
	# plt.legend()
	# plt.grid(True)
	# plt.show()


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

# def compare_methods(df):
#     # Calculate with least squares
#     n, x_sum, y_sum, xy_sum, x_squared_sum = calculate_least_squares_values(df)
#     m = calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum)
#     b = calculate_y_intercept(n, x_sum, y_sum, m)

#     # Calculate with gradient descent
#     theta0_gd, theta1_gd = gradient_descent(df['km'], df['price'])

#     print(f"Least Squares results: m = {m}, b = {b}")
#     print(f"Gradient Descent results: theta1 = {theta1_gd}, theta0 = {theta0_gd}")

#     # Optionally, calculate differences
#     diff_slope = abs(m - theta1_gd)
#     diff_intercept = abs(b - theta0_gd)
#     print(f"Difference in slope: {diff_slope}, Difference in intercept: {diff_intercept}")

def min_max_normalize(array):
	x_min = np.min(array)
	x_max = np.max(array)

	array_normalized = (array - x_min)/(x_max - x_min)
	return array_normalized



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
	df = pd.read_csv('data.csv')
	row, col = df.shape

	test1 = np.array([2.9, 2.3, 0.5]) #wanna use this as mileage which is used for predicting price
	test2 = np.array([3.2, 1.9, 1.4]) #these are the observed prices

	data_np_row = df['km'].values
	data_np_col = df['price'].values

	test_df = pd.DataFrame({
		'km': test1,
		'price': test2
	})

	# ret = gradient_descent(test_df)
	# print(ret)

	# compare_methods(df)
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
	# 	'km': [2.9, 2.3, 0.5],
	# 	'price': [3.2, 1.9, 1.4]
	# }

	# df = pd.DataFrame(data_statquest)

	# df_organic_chem_tutor = pd.DataFrame(data_organic_chem_tutor)
	# df_statquest = pd.DataFrame(data_statquest)
	# ret = gradient_descent(data_statquest)
	# print(ret)

	#normalize the data

	mileage_normalized = min_max_normalize(df['km'].values)
	price_normalized = min_max_normalize(df['price'].values)

	normalized_df = pd.DataFrame({
		'km': mileage_normalized,
		'price': price_normalized
	})

	# print(normalized_df)
	least_squares(normalized_df)

