#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import signal

from linear_regression import LinearRegression
from parse_regression import ParseRegression

 
def signal_handler(signum, frame):
	signame = signal.Signals(signum).name
	print(f'Signal handler called with signal {signame} ({signum}), exciting program.')
	sys.exit(0)



def main():
	parse_instance = ParseRegression()
	parse_instance.read_csv()
	parse_instance.split_dataset()
	parse_instance.save_datasets()

	linear_regression_instance = LinearRegression()
	linear_regression_instance.read_csv()
	linear_regression_instance.assign_mileage_and_price()
	result = linear_regression_instance.gradient_descent()
	
	linear_regression_instance.plot_linear_regression()
	linear_regression_instance.save_thetas(result)




if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	main()
