#!/opt/homebrew/bin/python3
import pandas as pd

class Parse:

	def __init__(self):
		self.train_data = 'data.csv'
		self.test_data = 'test.csv'



	def assign_mileage_and_price(self):
		self.mileage = self.min_max_normalize(self.df['km'].values)
		self.price = self.min_max_normalize(self.df['price'].values)



	def read_csv(self):
		try:
			df = pd.read_csv(self.train_data)
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



	def check_data_validity(self, df):
		# Ensure data has the expected columns and types
		if 'km' not in df or 'price' not in df:
			raise KeyError("Data must include 'km' and 'price' columns.")
		if not pd.api.types.is_numeric_dtype(df['km']) or not pd.api.types.is_numeric_dtype(df['price']):
			raise ValueError("Columns 'km' and 'price' must be numeric.")



	def get_percentage(self, prompt):
		while True:
			try:
				percentage = float(input(prompt))
				if 0 <= percentage <= 100:
					return percentage
				else:
					print("Percentage must be between 0 and 100.")
			except ValueError:
				print("Invalid input. Please enter a valid percentage.")



	def split_dataset(self):
		total_percentage = 100
		train_percentage = self.get_percentage("\n\nEnter the percentage for training data; \nSuggested values are: 80% for training, 20% for evaluating: ")
		evaluation_percentage = self.get_percentage("Enter the percentage for evaluation data: ")
		
		# Check if the sum of percentages exceeds 100
		if train_percentage + evaluation_percentage > total_percentage:
			print("The sum of percentages cannot exceed 100%. Please try again.")
			self.split_dataset()
		else:
			print(f"Percentage for training data: {train_percentage}%")
			print(f"Percentage for evaluation data: {evaluation_percentage}%")

		#call another functio which splits it into the given percentages
  
	def split_to_given_percentages(self):
		


def main():
	parse_instance = Parse()
	parse_instance.read_csv()
	parse_instance.split_dataset()
	#ask for a percentage of splitting dataset into train and evaluate

	# linear_regression_instance.assign_mileage_and_price()

	# result = linear_regression_instance.gradient_descent()
	
	# linear_regression_instance.plot_linear_regression()
	# linear_regression_instance.save_thetas(result)

	# linear_regression_instance.plot_mse_history()

if __name__ == '__main__':
	main()

	# def save_thetas(self, thetas):
	# 	with open('thetas.txt', 'w') as file:
	# 		print(f'{thetas[0]}, {thetas[1]}')
	# 		file.write(str(thetas[0]) + "\n")
	# 		file.write(str(thetas[1]) + "\n")
	# 	file.close()
		


	# # def signal_handler(signum, frame):
	# # 	signame = signal.Signals(signum).name
	# # 	print(f'Signal handler called with signal {signame} ({signum}), exciting program.')
	# # 	sys.exit(0)
