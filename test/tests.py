import os
import pandas as pd
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from command_line_parser import CommandLineParser
from dataset_processor import DatasetProcessor
from linear_regression import LinearRegression
from error_handler import ErrorHandler


def setup_module():
	os.makedirs('../data', exist_ok=True)



def teardown_module():
	if os.path.exists('../data'):
		for file in os.listdir('../data'):
			if file != 'data.csv':
				os.remove(os.path.join('../data', file))



def test_invalid_learning_rate():
	parser = CommandLineParser()
	sys.argv = ['train.py', '--learning_rate=0.7']
	try:
		parser.parse_arguments()
	except SystemExit as e:
		assert e.code == 1, "Expected SystemExit with code 1 for invalid learning rate"
	else:
		assert False, "SystemExit not raised for invalid learning rate"



def test_missing_file():
	processor = DatasetProcessor(80, 20, data_dir='/42_campus_jupyter')
	try:
		processor.process_data()
	except SystemExit as e:
		assert e.code == 1, "Expected SystemExit with code 1 for missing file"
	else:
		assert False, "SystemExit not raised for missing file"



def test_missing_columns():
	df = pd.DataFrame({'km': [1000, 2000, 3000]})
	temp_file = '../data/temp_data.csv'
	df.to_csv(temp_file, index=False)
	processor = DatasetProcessor(80, 20, data_file=temp_file)
	try:
		processor.process_data()
	except SystemExit as e:
		assert e.code == 1, "Expected SystemExit with code 1 for missing columns"
	else:
		assert False, "SystemExit not raised for missing columns"
	finally:
		os.remove(temp_file)



def test_non_numeric_columns():
	df = pd.DataFrame({'km': ['a', 'b', 'c'], 'price': [1000, 2000, 3000]})
	temp_file = '../data/temp_data.csv'
	df.to_csv(temp_file, index=False)
	processor = DatasetProcessor(80, 20, data_file=temp_file)
	try:
		processor.process_data()
	except SystemExit as e:
		assert e.code == 1, "Expected SystemExit with code 1 for non-numeric columns"
	else:
		assert False, "SystemExit not raised for non-numeric columns"
	finally:
		os.remove(temp_file)



def test_zero_variance():
	df = pd.DataFrame({'km': [1000, 1000, 1000], 'price': [1000, 1000, 1000]})
	temp_file = '../data/temp_training_data.csv'
	df.to_csv(temp_file, index=False)
	regression = LinearRegression(0.01, 0.0000001, data_file=temp_file)
	try:
		regression.run_linear_regression()
	except SystemExit as e:
		assert e.code == 1, "Expected SystemExit with code 1 for zero variance"
	else:
		assert False, "SystemExit not raised for zero variance"
	finally:
		os.remove(temp_file)



def main():
	setup_module()
	try:
		test_invalid_learning_rate()
		test_missing_file()
		test_missing_columns()
		test_non_numeric_columns()
		test_zero_variance()
	finally:
		teardown_module()

	print("\033[92mAll tests passed.")


if __name__ == '__main__':
	main()
