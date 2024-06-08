# Linear Regression Project

## Table of Contents
1. [Objective](#objective)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Code Description](#code-description)
    - [Command Line Parser](#command-line-parser)
    - [Dataset Processor](#dataset-processor)
    - [Error Handler](#error-handler)
    - [Linear Regression](#linear-regression)
    - [Plotting](#plotting)
    - [Prediction](#prediction)
    - [Training](#training)
6. [Contributing](#contributing)
7. [License](#license)

## Objective
The aim of this project is to introduce you to the basic concept behind machine learning. For this project, you will have to create a program that predicts the price of a car by using a linear function trained with a gradient descent algorithm. We will work on a precise example for the project, but once you’re done, you will be able to use the algorithm with any other dataset.

## Project Structure

├── data
│ ├── data.csv
│ ├── training_data.csv
│ ├── testing_data.csv
│ └── thetas.txt
├── src
│ ├── command_line_parser.py
│ ├── dataset_processor.py
│ ├── error_handler.py
│ ├── linear_regression.py
│ ├── plot_regression.py
│ ├── predict.py
│ └── train.py
└── README.md

bash


## Setup Instructions
1. **Clone the Repository**: Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/your-username/linear-regression-project.git
   cd linear-regression-project

    Install Dependencies: Ensure you have Python 3.7 or higher installed. Install the required dependencies using pip:

    bash

    pip install -r requirements.txt

    Prepare Data: Place your dataset in the data directory with the filename data.csv. Ensure the CSV contains km and price columns.

Usage

To run the linear regression training and prediction, use the following commands:
Training

bash

python src/train.py --plt=true --plt_mse=true --learning_rate=0.01 --convergence_threshold=0.0000001 --train_percentage=80 --test_percentage=20

Prediction

bash

python src/predict.py

Code Description
Command Line Parser

File: command_line_parser.py
This module handles the parsing of command line arguments using the argparse library. It allows customization of plotting options, learning rate, convergence threshold, and the percentage split for training and testing data.
Dataset Processor

File: dataset_processor.py
This module is responsible for reading the dataset, validating it, splitting it into training and testing sets based on the given percentages, and saving these sets as CSV files.
Error Handler

File: error_handler.py
This module provides a centralized way to handle and log errors. It includes specific handlers for different types of errors such as file not found, permission errors, and value errors.
Linear Regression

File: linear_regression.py
This module implements the linear regression model using gradient descent. It includes methods for reading the training data, normalizing it, performing gradient descent to optimize the model parameters, and calculating metrics such as MSE, RMSE, and MAE.
Plotting

File: plot_regression.py
This module uses matplotlib to plot the regression results and the Mean Squared Error (MSE) history during training. It provides visual insights into the performance and convergence of the model.
Prediction

File: predict.py
This script loads the trained model parameters and prompts the user to input the mileage of a car to predict its price. It uses the trained model to estimate the price and displays the result.
Training

File: train.py
This script orchestrates the entire training process. It parses command line arguments, processes the dataset, trains the linear regression model, optionally plots the results, and saves the trained model parameters.
Contributing

Contributions are welcome! If you have any improvements or suggestions, please open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for more details.
