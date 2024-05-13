#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import pandas as pd
import numpy as np
import csv
import argparse
import logging

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
		except (FileNotFoundError, PermissionError, IOError, pd.errors.EmptyDataError) as e:
			self.handle_file_error(e)

# 

	def handle_file_error(self, error):

		if isinstance(error, FileNotFoundError):
			print(f'File not found: {self.filename}')
		elif isinstance(error, PermissionError):
			print(f'Permission denied while trying to open the file: {self.filename}')
		elif isinstance(error, pd.errors.EmptyDataError):
			print(f'Datafile: {self.filename} is empty.')

		else:
			print(f'I/O error occured while reading the file: {self.filename}')
			print(f'Error details: {error}')



	def check_data_validity(self, df):
		if 'km' not in df or 'price' not in df:
			raise KeyError("Data must include 'km' and 'price' columns.")
		if not pd.api.types.is_numeric_dtype(df['km']) or not pd.api.types.is_numeric_dtype(df['price']):
			raise ValueError("Columns 'km' and 'price' must be numeric.")



	def split_dataset(self, train_percentage, test_percentage):
		if self.df is None:
			raise ValueError("Dataframe is empty. Make sure to read data first.")
		
		total_percentage = 100
		if self.train_percentage + self.test_percentage > total_percentage:
			print("The sum of percentages cannot exceed 100%. Please try again.")
			self.split_dataset()
		else:
			print(f"Percentage for training data: {self.train_percentage}%")
			print(f"Percentage for evaluation data: {self.test_percentage}%")

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

