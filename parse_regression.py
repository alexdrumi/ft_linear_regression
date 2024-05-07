#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import pandas as pd
import numpy as np
import csv



class ParseRegression:

	def __init__(self):
		self.filename = 'data.csv'
		self.train_data_name = 'training_data.csv'
		self.test_data_name = 'testing_data.csv'
		self.train_data = None
		self.test_data = None
		self.df = None
		self.df_len = 0
		self.train_percentage = 100
		self.test_percentage = 0



	def assign_mileage_and_price(self):
		self.mileage = self.min_max_normalize(self.df['km'].values)
		self.price = self.min_max_normalize(self.df['price'].values)



	def read_csv(self):
		try:
			df = pd.read_csv(self.filename)
			self.check_data_validity(df)
			self.df = df
			self.df_len = len(df['km'])
		except (FileNotFoundError, PermissionError, IOError) as e:
			self.handle_file_error(e)

# 

	def handle_file_error(self, error):

		if isinstance(error, FileNotFoundError):
			print(f'File not found: {self.filename}')
		elif isinstance(error, PermissionError):
			print(f'Permission denied while trying to open the file: {self.filename}')
		else:
			print(f'I/O error occured while reading the file: {self.filename}')
			print(f'Error details: {error}')



	def check_data_validity(self, df):
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
		self.train_percentage = self.get_percentage("\n\nEnter the percentage for training data; \nSuggested values are: 80% for training, 20% for evaluating: ")
		self.test_percentage = self.get_percentage("Enter the percentage for evaluation data: ")
		
		#check if the sum of percentages exceeds 100
		if self.train_percentage + self.test_percentage > total_percentage:
			print("The sum of percentages cannot exceed 100%. Please try again.")
			self.split_dataset()
		else:
			print(f"Percentage for training data: {self.train_percentage}%")
			print(f"Percentage for evaluation data: {self.test_percentage}%")

		#call another function which splits it into the given percentages
		self.split_to_given_percentages()



	def split_to_given_percentages(self):
		randomized_indices = np.random.permutation(self.df_len)
		train_index = int(self.train_percentage * self.df_len / 100)

		train_indices = randomized_indices[:train_index]
		test_indices = randomized_indices[train_index:]

		self.train_data = self.df.iloc[train_indices]
		self.test_data = self.df.iloc[test_indices]



	def save_datasets(self):
		self.train_data.to_csv(self.train_data_name, index=False)
		self.test_data.to_csv(self.test_data_name, index=False)