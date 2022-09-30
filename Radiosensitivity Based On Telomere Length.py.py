import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, auc, accuracy_score, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization


def combine_data(exploded_telos=None, all_patients_df=None, 
                 pred_obj='4 C means from individual telos',
                 timepoints_keep=['1 non irrad', '2 irrad @ 4 Gy']):
    
    if pred_obj == '4 C means from individual telos': 
        col_to_rename = 'telo means'
        col_to_keep = 'individual telomeres'
        target = '4 C telo means'
        
    elif pred_obj == '4 C means from telo means':
        col_to_rename = 'telo means'
        col_to_keep = 'telo means'
        target = '4 C telo means'
        
    elif pred_obj == '4 C # short telos from individual telos':
        col_to_rename = 'Q1'
        col_to_keep = 'individual telomeres'
        target = '4 C # short telos'
        
    elif pred_obj == '4 C # long telos from individual telos':
        col_to_rename = 'Q4'
        col_to_keep = 'individual telomeres'
        target = '4 C # long telos'
    
    # pulling out 4 C
    four_C = all_patients_df[all_patients_df['timepoint'] == '4 C'][['patient id', col_to_rename, 'timepoint']]
    four_C.rename(columns={col_to_rename: target}, inplace=True)

    if pred_obj == '4 C means from individual telos':
        # merging individual telomere data w/ 4 C telo means on patient id
        telo_data = (exploded_telos[exploded_telos['timepoint'] != '4 C']
                 .merge(four_C[[target, 'patient id']], on=['patient id']))
    
    elif pred_obj == '4 C means from telo means':
        telo_data = (all_patients_df[all_patients_df['timepoint'] != '4 C']
             .merge(four_C[[target, 'patient id']], on=['patient id']))
    
    elif pred_obj == '4 C # short telos from individual telos' or pred_obj == '4 C # long telos from individual telos':
        telo_data = (exploded_telos[exploded_telos['timepoint'] != '4 C']
                 .merge(four_C[[target, 'patient id']], on=['patient id']))

    telo_data = telo_data[['patient id', 'timepoint', col_to_keep, target]].copy()
    
    # timepoints of interest
    telo_data = telo_data[telo_data['timepoint'].isin(timepoints_keep)].copy()
    telo_data.reset_index(drop=True, inplace=True)
    return telo_data

exploded_telos_all_patients_df = pd.read_csv('../data/compiled patient data csv files/exploded_telos_all_patients_df.csv')
all_patients_df = pd.read_csv('../data/compiled patient data csv files/all_patients_df.csv')

# cleaning & combing data; retaining features of interest
telo_data = combine_data(exploded_telos=exploded_telos_all_patients_df, all_patients_df=all_patients_df)

# saving data to stylized table for manuscript

print(telo_data.shape)

example = telo_data.copy()
example.rename({'timepoint':'pre-therapy sample origin', 
                'individual telomeres':'individual telomeres (RFI)'}, axis=1, inplace=True)
example_8 = example[10:16].reset_index(drop=True)

# path=f'../graphs/paper figures/supp figs/view of precleaned individual telomere length dataframe.png'
# trp.render_mpl_table(example_8, col_width=4, path=path)

telo_test = telo_data.copy()

train_set, test_set = train_test_split(telo_test, test_size=0.2, shuffle=True, stratify=telo_test[['patient id', 'timepoint']])