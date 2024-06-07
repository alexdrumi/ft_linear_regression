import pandas as pd
import numpy as np
import logging
import sys
import os

from error_handler import ErrorHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
	"""
	manages the processing of datasets for machine learning purposes, handling
	tasks such as reading CSV data, checking validity, splitting into training
	and testing sets, and saving those datasets.
	"""
	def __init__(self, train_percentage, test_percentage, data_dir='../data'):
		"""
		initializes the DatasetProcessor with the path to the dataset and the split percentages.
		"""
		if data_dir is None:
			script_dir = os.path.dirname(__file__)
			data_dir = os.path.join(script_dir, '../data')
		self.data_dir = os.path.abspath(data_dir)
		self.filename = os.path.join(self.data_dir, 'data.csv')
		self.train_data_name = os.path.join(self.data_dir, 'training_data.csv')
		self.test_data_name = os.path.join(self.data_dir, 'testing_data.csv')
		self.train_data = None
		self.test_data = None
		self.df = None
		self.train_percentage = train_percentage
		self.test_percentage = test_percentage
		self.error_handler = ErrorHandler()



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

		total_percentage = 100
		if self.train_percentage + self.test_percentage > total_percentage:
			raise ValueError("The sum of train and test percentages cannot exceed 100%.")

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
		except Exception as e:
			self.error_handler.handle_error(e, self.filename)
