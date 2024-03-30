#!/opt/homebrew/bin/python3
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_graph_for_three(df):
    data_np_row_three = df['km'].head(3).values
    data_np_col_three = df['price'].head(3).values
    

    plt.plot(data_np_row_three, data_np_col_three, label='price estimation')
    plt.show()


if __name__ == "__main__":

    #maybe include guard in case failure
    df = pd.read_csv('data.csv')
    row, col = df.shape

    data_np_row = df['km'].values
    data_np_col = df['price'].values

    create_graph_for_three(df)
    # plt.plot(data_np_row, data_np_col, label='price estimation')
    # plt.show()


    # print(df)

    # df.T.plot()
    # print(f' after transpose: {df}')

    # df.plot()
    # plt.show()
    # print(df.iloc[0:row])