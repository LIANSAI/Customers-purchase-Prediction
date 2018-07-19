# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:39:55 2018

@author: Allen
"""


import pandas as pd
import numpy as np
from sklearn.metrics import  roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV


train_agg = pd.read_csv('file:///D:/数据/Kaggle/招行/train (1)/train_agg.csv',sep='\t')
train_flag = pd.read_csv('file:///D:/数据/Kaggle/招行/train (1)/train_flg.csv',sep='\t')
train_log = pd.read_csv('file:///D:/数据/Kaggle/招行/train (1)/train_log.csv',sep='\t')

test_agg = pd.read_csv('file:///D:/数据/Kaggle/招行/test/test_agg.csv',sep='\t')
test_log = pd.read_csv('file:///D:/数据/Kaggle/招行/test/test_log.csv',sep='\t')
test_flag = pd.read_csv('file:///D:/数据/Kaggle/招行/submit_sample.csv',sep='\t')

log = pd.read_csv('file:///D:/数据/Kaggle/招行/log_processed.csv')


del test_flag['RST']

test_flag['FLAG']=-1

agg = pd.concat([train_agg,test_agg],copy=False)
log = pd.concat([train_log,test_log],copy=False)
flg = pd.concat([train_flag,test_flag],copy=False)

data_raw = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)



def statisfeature(log,data):
    num_aggregations = {
            'day': ['min', 'max', 'mean','std'],
            'day_of_week': ['mean','min','max'],
            'hour': ['min','max', 'mean','std'],
            'minu': ['min', 'max', 'mean','std'],
        'click_times':['sum'],
        'day_minu':['max','min'],
        'next_minu':['mean','max'],
    
        }
    
    
    log_sta = log.groupby('USRID').agg({**num_aggregations}).reset_index()
    log_sta.columns = pd.Index([e[0] + "_" + e[1].upper() for e in log_sta.columns.tolist()])
    log_sta.rename(index=str,columns={'USRID_':'USRID'},inplace=True)
    data = pd.merge(data,log_sta,on='USRID',how='left')
    
    
    #按日期，按周几划分的点击次数，下一次点击间隔方差，斜度
    day_cilck = log.groupby(['USRID','day'])[['click_times']].agg('sum').reset_index().rename(index=str,columns={'click_times':'day_click'})
    day_cilck_sta = day_cilck.groupby(['USRID'])[['day_click']].agg(['skew','std']).reset_index().rename(index=str,columns={'day_click':'day_click_sta'})
    day_cilck_sta.columns = pd.Index([e[0] + "_" + e[1].upper() for e in day_cilck_sta.columns.tolist()])
    day_cilck_sta.rename(index=str,columns = {'USRID_': 'USRID'},inplace=True)
    data = pd.merge(data,day_cilck_sta,on='USRID',how='left')
    
    day_cilck = log.groupby(['USRID','day_of_week'])[['click_times']].agg('sum').reset_index().rename(index=str,columns={'click_times':'day_click'})
    day_cilck_sta = day_cilck.groupby(['USRID'])[['day_click']].agg(['skew','std']).reset_index().rename(index=str,columns={'day_click':'day_of_week_click_sta'})
    day_cilck_sta.columns = pd.Index([e[0] + "_" + e[1].upper() for e in day_cilck_sta.columns.tolist()])
    day_cilck_sta.rename(index=str,columns = {'USRID_': 'USRID'},inplace=True)
    data = pd.merge(data,day_cilck_sta,on='USRID',how='left')
    
    
    day_cilck = log.groupby(['USRID','day'])[['next_time']].agg('sum').reset_index().rename(index=str,columns={'click_times':'day_click'})
    day_cilck_sta = day_cilck.groupby(['USRID'])[['next_time']].agg(['skew','std']).reset_index().rename(index=str,columns={'day_click':'day_next_time_sta'})
    day_cilck_sta.columns = pd.Index([e[0] + "_" + e[1].upper() for e in day_cilck_sta.columns.tolist()])
    day_cilck_sta.rename(index=str,columns = {'USRID_': 'USRID'},inplace=True)
    data = pd.merge(data,day_cilck_sta,on='USRID',how='left')
    
    
    day_cilck = log.groupby(['USRID','day_of_week'])[['next_time']].agg('sum').reset_index().rename(index=str,columns={'click_times':'day_click'})
    day_cilck_sta = day_cilck.groupby(['USRID'])[['next_time']].agg(['std','skew']).reset_index().rename(index=str,columns={'day_click':'day_of_week_next_time_sta'})
    day_cilck_sta.columns = pd.Index([e[0] + "_" + e[1].upper() for e in day_cilck_sta.columns.tolist()])
    day_cilck_sta.rename(index=str,columns = {'USRID_': 'USRID'},inplace=True)
    data = pd.merge(data,day_cilck_sta,on='USRID',how='left')

    EVT_LBL_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
        
    data =  pd.merge(data,EVT_LBL_len,on='USRID',how='left')
    data =  pd.merge(data,EVT_LBL_set_len,on='USRID',how='left')
    
    return data



def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def corrtest(data):
    corrs = data.corr()
    threshold = 0.9
    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}
    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []
    
    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)
                
    cols_to_remove = list(set(cols_to_remove))
    print('Number of columns to remove: ', len(cols_to_remove))     
    return cols_to_remove


def clickdistribution(log): 
    #LEVEL1点击分布
    log_level1,new_col = one_hot_encoder(log[['USRID','level1']])
    cat_aggregations = {}
    for cat in new_col: cat_aggregations[cat] = ['sum']
    level1_agg = log_level1.groupby('USRID').agg({**cat_aggregations}).reset_index()
    level1_agg.columns = pd.Index(['level1_agg_' + e[0] + "_" + e[1].upper() for e in level1_agg.columns.tolist()])
    level1_agg.rename(index=str,columns = {'level1_agg_USRID_': 'USRID'},inplace=True)
    
    #LEVEL1点击概率分布
    log['count']=1
    f=log.groupby('USRID')[['count']].agg('sum').reset_index().rename(index=str,columns={'count':'total_click'})
    level1_agg = pd.merge(level1_agg,f,on='USRID',how='left')
    categorical_columns =  list(level1_agg.columns)[1:-1]
    for col in categorical_columns:
        level1_agg[col+'rate'] = level1_agg[col]/level1_agg.total_click    
   
    return level1_agg





if __name__ == "__main__":
    data = statisfeature(log,data_raw)
    print('Statistic feature done')   
   
    level1_distribution = clickdistribution(log)
    data =  pd.merge(data,level1_distribution,on='USRID',how='left')
    print('Level1 distribution done')
   
    corr= corrtest(data)
    data=data.drop(corr,axis=1)
    print('Drop high corr feature done')

    strfea=['V2','V3','V4','V5']
    data[strfea] = data[strfea].astype(str)
    data,col_new = one_hot_encoder(data)
    
    data.to_csv('stackingdata2.csv',index=False)








