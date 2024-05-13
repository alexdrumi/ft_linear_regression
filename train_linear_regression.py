#!/opt/homebrew/bin/python3
import sys
import signal

from linear_regression import LinearRegression
from parse_regression import ParseRegression
from argument_parser import ArgumentParser



def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} ({signum}), exiting program.')
	sys.exit(0)



def main(plot, plot_mse, learning_rate, convergence_treshold):
	arg_parser = ArgumentParser()
	args = arg_parser.parse()

	print(f"Plotting enabled: {args['plot']}, MSE plotting enabled: {args['plot_mse']}")

	# Instantiate and use the ParseRegression and LinearRegression with the parsed arguments
	parse_instance = ParseRegression()
	parse_instance.read_csv()
	parse_instance.split_dataset()
	parse_instance.save_datasets()

	linear_regression_instance = LinearRegression(args['learning_rate'], args['convergence_threshold'])
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()
	result = linear_regression_instance.gradient_descent()

	if args['plot']:
		linear_regression_instance.plot_linear_regression()
	if args['plot_mse']:
		linear_regression_instance.plot_mse_history()

	linear_regression_instance.save_thetas(result)
	linear_regression_instance.print_to_terminal()

	
	
	
	# parse_instance = ParseRegression()
	# parse_instance.read_csv()
	# parse_instance.split_dataset()
	# parse_instance.save_datasets()

	# linear_regression_instance = LinearRegression(learning_rate, convergence_treshold)
	# linear_regression_instance.read_csv()
	# linear_regression_instance.assign_mileage_and_price()
	# result = linear_regression_instance.gradient_descent()

	# if plot:
	# 	linear_regression_instance.plot_linear_regression()
	# if plot_mse:
	# 	linear_regression_instance.plot_mse_history()
	
	# linear_regression_instance.save_thetas(result)
	# linear_regression_instance.print_to_terminal()



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	main()
