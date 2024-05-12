#!/opt/homebrew/bin/python3
import sys
import signal
import argparse

from linear_regression import LinearRegression
from parse_regression import ParseRegression



def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} ({signum}), exiting program.')
	sys.exit(0)



def parse_arguments():

	usage = "train_linear_regression.py --pl=[true,false] --pl_mse=[true,false] --learning_rate=[float] --convergece_treshold=[float]\n"
	description = (
		"This script runs linear regression on a dataset with optional plotting. "
		"Specify --pl=true to enable plotting the regression results and --pl_mse=true "
		"to enable plotting the Mean Squared Error. By default, both plotting options are set to false."
		"Specify --learing_rate=some_float to change learning rate and --convergence_treshold=some_float to change the convergence_treshold.\n By default they are set [1e-2, 1e-7] respectively.\n"
	)

	parser = argparse.ArgumentParser(description=description, usage=usage)

	parser.add_argument('--plt', type=str, default='false', choices=['true', 'false'],
						help='Enable or disable the plot of the result of the regression model.')
	parser.add_argument('--plt_mse', type=str, default='false', choices=['true', 'false'],
						help='Enable or disable the plot of Mean Squared Error of the regression model.')
	parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate for gradient descent.')
	parser.add_argument('--convergence_threshold', type=float, default=0.0000001, help='Convergence threshold for stopping criterion.')


	if len(sys.argv) == 1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()
	plot = args.plt.lower() == 'true'
	plot_mse = args.plt_mse.lower() == 'true'
	learning_rate = args.learning_rate
	convergence_treshold = args.learning_rate


	return plot, plot_mse, learning_rate, convergence_treshold



def main(plot, plot_mse, learning_rate, convergence_treshold):
	parse_instance = ParseRegression()
	parse_instance.read_csv()
	parse_instance.split_dataset()
	parse_instance.save_datasets()

	linear_regression_instance = LinearRegression(learning_rate, convergence_treshold)
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()
	result = linear_regression_instance.gradient_descent()

	if plot:
		linear_regression_instance.plot_linear_regression()
	if plot_mse:
		linear_regression_instance.plot_mse_history()
	
	linear_regression_instance.save_thetas(result)
	linear_regression_instance.print_to_terminal()



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	
	plot, mse_plot, learning_rate, convergence_treshold = parse_arguments()
	print(f"Plotting enabled: {plot}, MSE plotting enabled: {mse_plot}")
	
	main(plot, mse_plot, learning_rate, convergence_treshold)
