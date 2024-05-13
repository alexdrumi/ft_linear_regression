#!/opt/homebrew/bin/python3


import sys
import argparse

class ArgumentParser:
	def __init__(self):
		self.parser = argparse.ArgumentParser(description=self.get_description(), usage=self.get_usage())
		self.add_arguments()



	def get_usage(self):
		return "train_linear_regression.py --plt=[true,false] --plt_mse=[true,false] --learning_rate=[float] --convergece_treshold=[float] --train_percentage=[float] --test_percentage=[float]\n"



	def get_description(self):
		return (
			"This script runs linear regression on a dataset with optional plotting. "
			"Specify --pl=true to enable plotting the regression results and --pl_mse=true "
			"to enable plotting the Mean Squared Error. By default, both plotting options are set to false."
			"Specify --learing_rate=some_float to change learning rate and --convergence_treshold=some_float to change the convergence_treshold.\n By default they are set [1e-2, 1e-7] respectively.\n"
			"Specify --train_percentage=[float] --test_percentage=[float] to chage the amount of percentages ot training/testing data. By default is 100/0 respectively."
		)



	def add_arguments(self):
		self.parser.add_argument('--plt', type=str, default='false', choices=['true', 'false'],
							help='Enable or disable the plot of the result of the regression model.')
		self.parser.add_argument('--plt_mse', type=str, default='false', choices=['true', 'false'],
							help='Enable or disable the plot of Mean Squared Error of the regression model.')
		self.parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate for gradient descent.')
		self.parser.add_argument('--convergence_threshold', type=float, default=0.0000001, help='Convergence threshold for stopping criterion.')
		self.parser.add_argument('--train_percentage', type=float, default=100, help='The amount of percentages of the original dataset for training the model.')
		self.parser.add_argument('--test_percentage', type=float, default=0, help='The amount of percentages of the original dataset for testing the model.')



	def parse_arguments(self):
		self.add_arguments()

		if len(sys.argv) == 1:
			self.parser.print_help(sys.stderr)
			sys.exit(1)

		args = self.parser.parse_args()

		plot = args.plt.lower() == 'true'
		plot_mse = args.plt_mse.lower() == 'true'
		learning_rate = args.learning_rate
		convergence_treshold = args.convergence_threshold
		test_percentage = args.test_percentage
		train_percentage = args.train_percentage

		return plot, plot_mse, learning_rate, convergence_treshold, train_percentage, test_percentage
