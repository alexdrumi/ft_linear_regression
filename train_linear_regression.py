#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import signal

from linear_regression import LinearRegression
from dataset_processor import DatasetProcessor
from command_line_parser import CommandLineParser
from plot_regression import PlotRegression



def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} -{signum}-, exiting program.')
	sys.exit(0)



def main():
	#parse from command line
	arg_parser = CommandLineParser()
	plot, plot_mse, learning_rate, convergence_treshold, train_percentage, test_percentage = arg_parser.parse_arguments()

	#preprocess data and split into datasets
	parse_instance = DatasetProcessor(train_percentage, test_percentage)
	parse_instance.read_csv()
	parse_instance.split_dataset(train_percentage, test_percentage)
	parse_instance.save_datasets()

	#use splitted datasets with specified learning rate and convergence treshold
	linear_regression_instance = LinearRegression(learning_rate, convergence_treshold)
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()
	result = linear_regression_instance.gradient_descent()

	#if plot or plot_mse options were requested, plot then
	plot_regression_instance = PlotRegression(linear_regression_instance.iterations,
	linear_regression_instance.theta0, 
	linear_regression_instance.theta1, 
	linear_regression_instance.mileage, 
	linear_regression_instance.price,
	linear_regression_instance.mse_history)

	if plot:
		plot_regression_instance.plot_linear_regression()
	if plot_mse:
		plot_regression_instance.plot_mse_history()

	#save results and print it to display
	linear_regression_instance.save_thetas(result)
	linear_regression_instance.print_to_terminal()



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	main()
