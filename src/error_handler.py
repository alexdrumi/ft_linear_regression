#!/opt/homebrew/bin/python3
import logging
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ErrorHandler:
	"""
	handles exceptions and logs errors with appropriate messages.
	"""
	def __init__(self):
		self.error_handlers = [
			(FileNotFoundError, "File not found: {filename}, exiting program."),
			(PermissionError, "Permission denied while trying to open the file: {filename}, exiting program."),
			(pd.errors.EmptyDataError, "Datafile is empty: {filename}, exiting program."),
			(KeyError, "Missing necessary columns in the data: {error}, exiting program."),
			(ValueError, "Data error: {error}, exiting program."),
			(IOError, "I/O error occurred while reading the file: {filename}\nError details: {error}, exiting program."),
			(Exception, "An unexpected error occurred: {error}, exiting program.")
		]



	def handle_error(self, error, filename=None):
		for error_type, message in self.error_handlers:
			if isinstance(error, error_type):
				formatted_message = message.format(filename=filename, error=error)
				logging.error(formatted_message)
				sys.exit(1)



	def log_message(self, message):
		logging.error(message)
		sys.exit(1)
