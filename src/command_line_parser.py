import sys
import argparse
from error_handler import ErrorHandler

class CommandLineParser:
	def __init__(self):
		self.error_handler = ErrorHandler()
		self.parser = argparse.ArgumentParser(
			description=self.get_description(),
			usage=self.get_usage(),
			formatter_class=argparse.RawTextHelpFormatter
		)
		self.add_arguments()

	def get_usage(self):
		usage_text = (
			"\033[1;33m\n\ntrain_linear_regression.py\033[0m"
			" --plt=[\033[1;32mtrue,false\033[0m]"
			" --plt_mse=[\033[1;32mtrue,false\033[0m]"
			" --learning_rate=[\033[1;34mfloat\033[0m]"
			" --convergence_threshold=[\033[1;34mfloat\033[0m]"
			" --train_percentage=[\033[1;34mfloat\033[0m]"
			" --test_percentage=[\033[1;34mfloat\033[0m]\n"
		)
		return usage_text

	def get_description(self):
		description_text = (
			"\033[1;36mThis script runs linear regression on a dataset with optional plotting.\033[0m\n"
			"Specify \033[1;32m--plt=true\033[0m to enable plotting the regression results and \033[1;32m--plt_mse=true\033[0m "
			"to enable plotting the Mean Squared Error. By default, both plotting options are set to false.\n"
			"Specify \033[1;34m--learning_rate=float\033[0m to change learning rate and \033[1;34m--convergence_threshold=float\033[0m to change the convergence threshold.\n"
			"By default they are set [\033[1;35m1e-2\033[0m, \033[1;35m1e-7\033[0m] respectively.\n"
			"Specify \033[1;34m--train_percentage=float\033[0m --\033[1;34mtest_percentage=float\033[0m to change the amount of percentages of training/testing data. By default is 100/0 respectively."
		)
		return description_text

	def add_arguments(self):
		self.parser.add_argument('--plt', type=str, default='false', choices=['true', 'false'],
								 help='Enable or disable the plot of the result of the regression model.\n')
		self.parser.add_argument('--plt_mse', type=str, default='false', choices=['true', 'false'],
								 help='Enable or disable the plot of Mean Squared Error of the regression model.\n')
		self.parser.add_argument('--learning_rate', type=float, default=0.01,
								 help='Initial learning rate for gradient descent.\n')
		self.parser.add_argument('--convergence_threshold', type=float, default=0.0000001,
								 help='Convergence threshold for stopping criterion.\n')
		self.parser.add_argument('--train_percentage', type=float, default=100,
								 help='The amount of percentages of the original dataset for training the model.\n')
		self.parser.add_argument('--test_percentage', type=float, default=0,
								 help='The amount of percentages of the original dataset for testing the model.\n')

	def parse_arguments(self):
		if len(sys.argv) > 7:
			self.parser.print_help(sys.stderr)
			sys.exit(1)

		args = self.parser.parse_args()

		plot = args.plt.lower() == 'true'
		plot_mse = args.plt_mse.lower() == 'true'

		learning_rate = args.learning_rate
		convergence_treshold = args.convergence_threshold
		test_percentage = args.test_percentage
		train_percentage = args.train_percentage

		#raise problems here w error handler
		try:
			if train_percentage + test_percentage > 100:
				raise ValueError("The sum of train and test percentages cannot exceed 100%.")
			if train_percentage < 0 or train_percentage > 100 or test_percentage < 0 or test_percentage > 100:
				raise ValueError("The percentages have to be within the range of 0 <= percentage <= 100.")
			if learning_rate < 1e-6 or learning_rate > 2e-1:
				raise ValueError("Learning rate has to be within the range of 1e-6 <= learning rate <= 2e-1.")
			if convergence_treshold < 1e-7 or convergence_treshold > 1e-5:
				raise ValueError("The convergence threshold has to be in the range of 1e-7 <= convergence threshold <= 1e-5.")
		except ValueError as e:
			self.error_handler.log_message(str(e))

		return plot, plot_mse, learning_rate, convergence_treshold, train_percentage, test_percentage
