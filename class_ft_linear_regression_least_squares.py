
class LinearRegressionLeastSquared:

    
	def calculate_least_squares_values(self, df):
		n = df.shape[0]
		xy = df['km'] * df['price']
		x_squared = df['km'] * df['km']

		x_sum = df['km'].sum()
		y_sum = df['price'].sum()
		xy_sum = xy.sum()
		x_squared_sum = x_squared.sum()

		return n, x_sum, y_sum, xy_sum, x_squared_sum



	def calculate_slope(self, n, x_sum, y_sum, xy_sum, x_squared_sum):
		m_nominator = (n * xy_sum) - (x_sum * y_sum)
		m_denominator = (n * x_squared_sum) - (x_sum ** 2)
		m = m_nominator / m_denominator

		return m



	def calculate_y_intercept(self, n, x_sum, y_sum, m):
		b_nominator = y_sum - (m * x_sum)
		b_denominator = n
		b = b_nominator / b_denominator
		
		return b



	def least_squares(self, df):
		n, x_sum, y_sum, xy_sum, x_squared_sum = self.calculate_least_squares_values(df)
		
		#we would like to find the slope for m
		m = self.calculate_slope(n, x_sum, y_sum, xy_sum, x_squared_sum)
		
		#we would like to find b (y intercept)
		b = self.calculate_y_intercept(n, x_sum, y_sum, m)

		print(f'{m} is slope, {b} is intercept')
		#these are the values for the fitted line
		x_values = df['km']
		y_values = m * x_values + b

		theta0, theta1 = self.gradient_descent(df)

		# Generate a range of mileage values for plotting
		x_range = np.linspace(df['km'].min(), df['km'].max(), 100)

		# Predicted values from least squares
		y_least_squares = m * x_range + b

		# Predicted values from gradient descent
		y_gradient_descent = theta1 * x_range + theta0

