import signal
import sys

from linear_regression import LinearRegression
from dataset_processor import DatasetProcessor
from command_line_parser import CommandLineParser
from plot_regression import PlotRegression

def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} -{signum}-, exiting program.')
	sys.exit(0)



def main():
	arg_parser = CommandLineParser()
	plot, plot_mse, learning_rate, convergence_treshold, train_percentage, test_percentage = arg_parser.parse_arguments()

	parse_instance = DatasetProcessor(train_percentage, test_percentage)
	parse_instance.process_data()

	linear_regression_instance = LinearRegression(learning_rate, convergence_treshold)
	result = linear_regression_instance.run_linear_regression()

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

	linear_regression_instance.save_thetas(result)
	linear_regression_instance.print_to_terminal()



if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	main()
