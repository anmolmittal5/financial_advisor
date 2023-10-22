import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
import datetime
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
# # from tensorflow.keras import regularizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

'''
stocks = pd.read_csv('C:/Users/kaush/Documents/CDS/sp500_returns_with_dates.csv')
bonds = pd.read_csv('C:/Users/kaush/Documents/CDS/us_bond_data_cleaned.csv')
dollars = pd.read_csv('C:/Users/kaush/Documents/CDS/us_dollar_returns2022 onwards.csv')



def prepare_df(stocks, bonds, dollars):
    stocks['Asset'] = 'Stocks'
    bonds['Asset'] = 'Bonds'
    dollars['Asset'] = 'Cash'
    stocks = stocks.bfill()
    bonds = bonds.bfill()
    dollars = dollars.bfill()
    stocks['Date'] = pd.to_datetime(stocks['Date'])
    bonds['Date'] = pd.to_datetime(bonds['Date'])
    dollars['Date'] = pd.to_datetime(dollars['Date'], format="%d/%m/%Y")
    return stocks, bonds, dollars

stocks, bonds, dollars = prepare_df(stocks, bonds, dollars)


def merge_dfs(df1, df2, df3):
    asset1 = df1['Asset'].unique()[0]
    asset2 = df2['Asset'].unique()[0]
    asset3 = df3['Asset'].unique()[0]
    new_col1 = asset1+'_per_returns'
    new_col2 = asset2+'_per_returns'
    new_col3 = asset3+'_per_returns'
    df1_new = df1.rename(columns={'Returns': 'Returns_'+asset1, 'Asset': 'Asset_'+asset1})
    df2_new = df2.rename(columns={'Returns': 'Returns_'+asset2, 'Asset': 'Asset_'+asset2})
    df3_new = df3.rename(columns={'Returns': 'Returns_'+asset3, 'Asset': 'Asset_'+asset3})
    df1_new = df1_new.set_index('Date')
    df2_new = df2_new.set_index('Date')
    df3_new = df3_new.set_index('Date')
    # d_new.head()
    concat_df = pd.concat([df1_new,df2_new, df3_new], axis=1)
    concat_df = concat_df.ffill()
    concat_df = concat_df.bfill()
    concat_df['total_returns'] = concat_df['Returns_'+asset1] + concat_df['Returns_'+asset2] + concat_df['Returns_'+asset3]
    concat_df[new_col1] = concat_df['Returns_'+asset1]*100/concat_df['total_returns']
    concat_df[new_col2] = concat_df['Returns_'+asset2]*100/concat_df['total_returns']
    concat_df[new_col3] = concat_df['Returns_'+asset3]*100/concat_df['total_returns']
    return concat_df

merged_df = merge_dfs(stocks, bonds, dollars)


def create_input_sequences(data, sequence_length):
    # Extract relevant columns
    returns = data['total_returns'][-7:].values
    stock_percentage = data['Stocks_per_returns'][-7:].values
    bond_percentage = data['Bonds_per_returns'][-7:].values
    dollar_percentage = data['Cash_per_returns'][-7:].values

    # Combine the input features and target variables
    X_ts7 = np.column_stack((returns, stock_percentage, bond_percentage, dollar_percentage))

    # Normalize the data
    scaler = StandardScaler()
    X_ts7 = scaler.fit_transform(X_ts7)

    return X_ts7

data = merged_df
sequence_length = 7  # You want to predict the next 7 days
input_sequence = create_input_sequences(data, sequence_length)
input_sequence = input_sequence.reshape(1, sequence_length, input_sequence.shape[1])

'''

input_sequence = [[[-1.23548854, -0.34773504,  0.37016103, -0.3988336 ],
        [ 0.58636271, -0.39537122,  0.42780971, -0.46944507],
        [-0.1100707 , -0.64677616,  0.57999954, -0.49210003],
        [-0.63707234,  2.43698303, -2.44224021,  2.44542922],
        [-0.99474293, -0.35575759,  0.38393767, -0.42008887],
        [ 1.85269229, -0.34842693,  0.34988244, -0.35125468],
        [ 0.53831951, -0.34291609,  0.33044981, -0.31370698]]]

# with open('models/asset_allocation.bin', 'rb') as model_file:
#     model_data = pickle.load(model_file)

# 1. Load the model architecture (JSON) from the binary file
# with open('models/asset_allocation_new.bin', 'rb') as model_file:
with open('models/asset_allocation_new.bin', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model_json = model_data['model_json']

# 2. Load the weights from the HDF5 file
loaded_model = tf.keras.models.model_from_json(model_json)
# loaded_model.load_weights('models/asset_allocation_weights_new.h5')
loaded_model.load_weights('models/asset_allocation_weights_new.h5')

def pred_allocations(model,input_seq, n):
    predicted_percentages = model.predict(input_sequence)
    return predicted_percentages

def get_percentage_allocations(model, input_seq, n):
    predicted_percentages = pred_allocations(model, input_seq, n)
    total = sum(predicted_percentages[0][n])
    new_total = 0
    pred_dict = {}
    pred_dict['pred_stocks'] = predicted_percentages[0][n][0]
    pred_dict['pred_bonds'] = predicted_percentages[0][n][1]
    pred_dict['pred_cash'] = predicted_percentages[0][n][2]
    if total>0:
        for key in pred_dict.keys():
            if pred_dict[key]>0:
                new_total += pred_dict[key]
            else:
                pred_dict[key] = 0
    else:
        for key in pred_dict.keys():
            if pred_dict[key]<0:
                new_total += pred_dict[key]
            else:
                pred_dict[key] = 0

    pred_dict["pred_stocks"] = pred_dict["pred_stocks"]*100/new_total
    pred_dict["pred_bonds"] = pred_dict["pred_bonds"]*100/new_total
    pred_dict["pred_cash"] = pred_dict["pred_cash"]*100/new_total

    return pred_dict
    

# pred_dict = get_percentage_allocations(loaded_model, input_sequence, 0)
# print(f'Percentage of Stocks Allocation = {pred_dict["pred_stocks"]}\nPercentage of Bonds Allocation = {pred_dict["pred_bonds"]}\nPercentage of Cash Allocation = {pred_dict["pred_cash"]}')

def allocations_personal_info(input_seq, age, risk_tolerance, investment_goal, income_level, expenses_level, knowledge_experience, family_situation, n=0):
    model = loaded_model
    pred_dict = get_percentage_allocations(model, input_seq, n)
# print(f'Percentage of Stocks Allocation = {pred_dict["pred_stocks"]}\nPercentage of Bonds Allocation = {pred_dict["pred_bonds"]}\nPercentage of Cash Allocation = {pred_dict["pred_cash"]}')
    shares_pre_calculated = pred_dict['pred_stocks']
    bonds_pre_calculated = pred_dict['pred_bonds']
    cash_pre_calculated = pred_dict['pred_cash']
    if age > 50:
        shares_allocation_age = 0.5
        bonds_allocation_age = 0.4
        cash_allocation_age = 0.1
    elif age < 30:
        shares_allocation_age = 0.7
        bonds_allocation_age = 0.2
        cash_allocation_age = 0.1
    else:
        shares_allocation_age = 0.6
        bonds_allocation_age = 0.3
        cash_allocation_age = 0.1

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_age
    bonds_pre_calculated *= bonds_allocation_age
    cash_pre_calculated *= cash_allocation_age

    if risk_tolerance == 'high':
        shares_allocation_rt = 0.7
        bonds_allocation_rt = 0.2
        cash_allocation_rt = 0.1
    elif risk_tolerance == 'low':
        shares_allocation_rt = 0.4
        bonds_allocation_rt = 0.4
        cash_allocation_rt = 0.2

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_rt
    bonds_pre_calculated *= bonds_allocation_rt
    cash_pre_calculated *= cash_allocation_rt

    if investment_goal == 'short-term':
        shares_allocation_ig = 0.3
        bonds_allocation_ig = 0.5
        cash_allocation_ig = 0.2
    elif investment_goal == 'long-term':
        shares_allocation_ig = 0.7
        bonds_allocation_ig = 0.2
        cash_allocation_ig = 0.1

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_ig
    bonds_pre_calculated *= bonds_allocation_ig
    cash_pre_calculated *= cash_allocation_ig

    if income_level == 'high' and expenses_level == 'low':
        shares_allocation_il = 0.6
        bonds_allocation_il = 0.3
        cash_allocation_il = 0.1
    elif income_level == 'low' and expenses_level == 'high':
        shares_allocation_il = 0.4
        bonds_allocation_il = 0.5
        cash_allocation_il = 0.1

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_il
    bonds_pre_calculated *= bonds_allocation_il
    cash_pre_calculated *= cash_allocation_il

    if knowledge_experience == 'low':
        shares_allocation_ke = 0.4
        bonds_allocation_ke = 0.4
        cash_allocation_ke = 0.2
    elif knowledge_experience == 'high':
        shares_allocation_ke = 0.6
        bonds_allocation_ke = 0.3
        cash_allocation_ke = 0.1

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_ke
    bonds_pre_calculated *= bonds_allocation_ke
    cash_pre_calculated *= cash_allocation_ke


    if family_situation == 'single_no_children':
        shares_allocation_fs = 0.6
        bonds_allocation_fs = 0.3
        cash_allocation_fs = 0.1
    elif family_situation == 'married_young_children':
        shares_allocation_fs = 0.4
        bonds_allocation_fs = 0.5
        cash_allocation_fs = 0.1

    # Update pre-calculated allocations
    shares_pre_calculated *= shares_allocation_fs
    bonds_pre_calculated *= bonds_allocation_fs
    cash_pre_calculated *= cash_allocation_fs

    total_allocation = shares_pre_calculated + bonds_pre_calculated + cash_pre_calculated
    shares_pre_calculated /= total_allocation
    bonds_pre_calculated /= total_allocation
    cash_pre_calculated /= total_allocation

    allocation_dict = {'Shares': shares_pre_calculated*100, 'Bonds':bonds_pre_calculated*100, 'Cash':cash_pre_calculated*100}

    return allocation_dict

def get_asset_allocations(age, risk_tolerance, investment_goal, income_level, expenses_level, knowledge_experience, family_situation):
    input_sequence = [[[-1.23548854, -0.34773504,  0.37016103, -0.3988336 ],
        [ 0.58636271, -0.39537122,  0.42780971, -0.46944507],
        [-0.1100707 , -0.64677616,  0.57999954, -0.49210003],
        [-0.63707234,  2.43698303, -2.44224021,  2.44542922],
        [-0.99474293, -0.35575759,  0.38393767, -0.42008887],
        [ 1.85269229, -0.34842693,  0.34988244, -0.35125468],
        [ 0.53831951, -0.34291609,  0.33044981, -0.31370698]]]
    
    final_dict = allocations_personal_info(input_sequence, age, risk_tolerance, investment_goal, income_level, expenses_level, knowledge_experience, family_situation)

    return final_dict


# fd = get_asset_allocations(25, 'high', 'long-term', 'low', 'high', 'low', 'single_no_children')
# print(fd)


