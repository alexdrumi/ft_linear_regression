#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import pandas as pd
import numpy as np
import logging
import sys


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
	"""
	Manages the processing of datasets for machine learning purposes, handling
	tasks such as reading CSV data, checking validity, splitting into training
	and testing sets, and saving those datasets.
	"""
	def __init__(self, train_percentage, test_percentage, filename='data.csv', train_data_name='training_data.csv', test_data_name='testing_data.csv'):
		"""
		Initializes the DatasetProcessor with the path to the dataset and the split percentages.
		"""
		self.filename = filename
		self.train_data_name = train_data_name
		self.test_data_name = test_data_name
		self.train_data = None
		self.test_data = None
		self.df = None
		self.train_percentage = train_percentage
		self.test_percentage = test_percentage



	def read_csv(self):
		try:
			df = pd.read_csv(self.filename)
			self.check_data_validity(df)
			self.df = df
			self.df_len = len(df['km'])
		except (FileNotFoundError, PermissionError, IOError, pd.errors.EmptyDataError, KeyError, ValueError) as e:
			self.handle_file_error(e)



	def check_data_validity(self, df):
		if 'km' not in df or 'price' not in df:
			raise KeyError("Data must include 'km' and 'price' columns.")
		if not pd.api.types.is_numeric_dtype(df['km']) or not pd.api.types.is_numeric_dtype(df['price']):
			raise ValueError("Columns 'km' and 'price' must be numeric.")



	def handle_file_error(self, error):

		if isinstance(error, FileNotFoundError):
			sys.exit(f'File not found: {self.filename}, exiting program.')
		elif isinstance(error, PermissionError):
			sys.exit(f'Permission denied while trying to open the file: {self.filename}, exiting program.')
		elif isinstance(error, ValueError):
			sys.exit(f'Incorrect values in {self.filename}, exiting program.')
		elif isinstance(error, pd.errors.EmptyDataError):
			sys.exit(f'Datafile: {self.filename} is empty. Exciting program.')
		else:
			sys.exit(f'I/O error occurred while reading the file: {self.filename}\nError details: {error}, exiting program.')



	def split_dataset(self):
		if self.df is None:
			raise ValueError("Dataframe is empty. Make sure to read data first.")

		total_percentage = 100
		if self.train_percentage + self.test_percentage > total_percentage:
			logging.error("The sum of percentages cannot exceed 100%. Exiting program.")
			sys.exit(1)
		
		self.split_to_given_percentages()
		self.validate_datasets()



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



	def check_for_missing_values(self, data, name):
		if data is not None and data.isnull().values.any():
			raise ValueError(f"There might be missing or incorrect values in the {name} dataset.")



	def validate_datasets(self):
		self.check_for_missing_values(self.train_data, "train")
		self.check_for_missing_values(self.test_data, "test")



	def process_data(self):
		try:
			logging.info("Starting data processing...")
			self.read_csv()
			self.split_dataset()
			self.save_datasets()
		except (ValueError) as e:
			# loggig.if gotta log these etc
		



''' tomorrow for a more elegant error handling
class DatasetProcessor:
	"""
	Manages the processing of datasets for machine learning purposes, handling
	tasks such as reading CSV data, checking validity, splitting into training
	and testing sets, and saving those datasets.
	"""
	def __init__(self, train_percentage, test_percentage, filename='data.csv', train_data_name='training_data.csv', test_data_name='testing_data.csv'):
		self.filename = filename
		self.train_data_name = train_data_name
		self.test_data_name = test_data_name
		self.train_data = None
		self.test_data = None
		self.df = None
		self.train_percentage = train_percentage
		self.test_percentage = test_percentage

	def read_csv(self):
		df = pd.read_csv(self.filename)
		self.check_data_validity(df)
		self.df = df
		self.df_len = len(df['km'])

	def check_data_validity(self, df):
		if 'km' not in df or 'price' not in df:
			raise KeyError("Data must include 'km' and 'price' columns.")
		if not pd.api.types.is_numeric_dtype(df['km']) or not pd.api.types.is_numeric_dtype(df['price']):
			raise ValueError("Columns 'km' and 'price' must be numeric.")

	def split_dataset(self):
		if self.df is None:
			raise ValueError("Dataframe is empty. Make sure to read data first.")
		if self.train_percentage + self.test_percentage > 100:
			raise ValueError("The sum of percentages cannot exceed 100%.")
		self.split_to_given_percentages()
		self.validate_datasets()

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

	def check_for_missing_values(self, data, name):
		if data.isnull().values.any():
			raise ValueError(f"There might be missing or incorrect values in the {name} dataset.")

	def validate_datasets(self):
		self.check_for_missing_values(self.train_data, "train")
		self.check_for_missing_values(self.test_data, "test")

	def process_data(self):
		logging.info("Starting data processing...")
		try:
			self.read_csv()
			self.split_dataset()
			self.save_datasets()
			logging.info("Data processing completed successfully.")
		except (FileNotFoundError, PermissionError) as e:
			logging.error(f"File error: {e}")
		except (KeyError, ValueError) as e:
			logging.error(f"Data validation error: {e}")
		except pd.errors.EmptyDataError:
			logging.error(f"Datafile: {self.filename} is empty. Exiting program.")
		except IOError as e:
			logging.error(f"I/O error: {e}")
		except Exception as e:
			logging.error(f"Unexpected error: {e}")
		sys.exit(1)
'''