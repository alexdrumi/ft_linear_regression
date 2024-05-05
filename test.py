#!/opt/homebrew/bin/python3

import math


def partial_sum(nth):
	return (5**nth)


if __name__ == '__main__':
	result = 0
	for index in range(1, 9):
		print(index)
		result += partial_sum(index)
	print(result)