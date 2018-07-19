# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:33:05 2018

@author: Allen
"""



import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import  roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

train_agg = pd.read_csv('file:///D:/数据/Kaggle/招行/train (1)/train_agg.csv',sep='\t')
train_flag = pd.read_csv('file:///D:/数据/Kaggle/招行/train (1)/train_flg.csv',sep='\t')
log = pd.read_csv('file:///D:/数据/Kaggle/招行/log_processed.csv')

test_agg = pd.read_csv('file:///D:/数据/Kaggle/招行/test/test_agg.csv',sep='\t')
test_flag = pd.read_csv('file:///D:/数据/Kaggle/招行/submit_sample.csv',sep='\t')

del test_flag['RST']
test_flag['FLAG']=-1

agg = pd.concat([train_agg,test_agg],copy=False)
flg = pd.concat([train_flag,test_flag],copy=False)

data_raw = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)



###############1. Feature 重度用户 ################
def basicfeature(log,data):
    f = log[['USRID','day']].drop_duplicates()
    f1 = f.groupby(['USRID'])[['day']].agg('count').reset_index().rename(index=str,columns={'day':'total_use_day'})
    data =  pd.merge(data,f1,on='USRID',how='left')
    
    EVT_LBL_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    data =  pd.merge(data,EVT_LBL_len,on='USRID',how='left')
    data =  pd.merge(data,EVT_LBL_set_len,on='USRID',how='left')
    del EVT_LBL_len,EVT_LBL_set_len
    gc.collect()
    
    data['mean_day_click'] = data.EVT_LBL_len/data.total_use_day
    #看过几个LEVEL1
    f = log[['USRID','level1']].drop_duplicates()
    f1 = f.groupby(['USRID'])[['level1']].agg('count').reset_index().rename(index=str,columns={'level1':'level1_set_len'})
    data=  pd.merge(data,f1,on='USRID',how='left')
    
    #LEVEL1max mean点击
    f = log.groupby(['USRID'])[['level1']].agg('count').reset_index().rename(index=str,columns={'level1':'max_level1'})
    num_aggregations = {
            'max_level1': ['max', 'mean']}
    
    f1 = f.groupby('USRID').agg({**num_aggregations}).reset_index()
    f1.columns = pd.Index([e[0] + "_" + e[1].upper() for e in f1.columns.tolist()])
    f1.rename(index=str,columns={'USRID_':'USRID'},inplace=True)
    data =  pd.merge(data,f1,on='USRID',how='left')
    
    #看过几个LEVEL2
    f = log[['USRID','level2']].drop_duplicates()
    f1 = f.groupby(['USRID'])[['level2']].agg('count').reset_index().rename(index=str,columns={'level2':'level2_set_len'})
    data =  pd.merge(data,f1,on='USRID',how='left')
    
    #LEVEL2max mean点击
    f = log.groupby(['USRID'])[['level2']].agg('count').reset_index()
    num_aggregations = {
            'level2': ['max', 'mean']}
    
    f1 = f.groupby('USRID').agg({**num_aggregations}).reset_index()
    f1.columns = pd.Index([e[0] + "_" + e[1].upper() for e in f1.columns.tolist()])
    f1.rename(index=str,columns={'USRID_':'USRID'},inplace=True)
    data =  pd.merge(data,f1,on='USRID',how='left')
    
    
    #一个l2看过几个l3
    data['level3/level2'] = data.EVT_LBL_set_len/data.level2_set_len
    
    #最大几号看过
    f = log.groupby('USRID')[['day']].agg('max').reset_index().rename(index=str,columns={'day':'max_day'})
    data = pd.merge(data,f,on='USRID',how='left')
    
    return data

   
def extrafeature(log,data):    
    #周末点击占比
    f = log.groupby('USRID')[['is_weekend']].agg('sum').reset_index().rename(index=str,columns={'is_weekend':'is_weekend'})
    data =  pd.merge(data,f,on='USRID',how='left')
    data['is_weekend_ratio'] = data.is_weekend/data.EVT_LBL_len
    data['is_weekend_ratio_mean'] = data.is_weekend/data.mean_day_click
    
    #周三点击占比
    f = log.groupby('USRID')[['is_Wednesday']].agg('sum').reset_index()
    data =  pd.merge(data,f,on='USRID',how='left')
    data['is_Wednesday_ratio'] = data.is_Wednesday/data.EVT_LBL_len
    data['is_Wednesday_ratio_mean'] = data.is_Wednesday/data.mean_day_click
    
    #Late hour占比
    f = log.groupby('USRID')[['is_late_hour']].agg('sum').reset_index()
    data =  pd.merge(data,f,on='USRID',how='left')
    data['is_late_hour_ratio'] = data.is_late_hour/data.EVT_LBL_len
    data['is_late_hour_ratio_mean'] = data.is_late_hour/data.mean_day_click
    
    f = log.groupby('USRID')[['click_within_onehour']].agg('sum').reset_index()
    data =  pd.merge(data,f,on='USRID',how='left')
    
    #最后2天点击占比
    last_2_day = log[log.day>=30]
    f = last_2_day.groupby('USRID')[['click_times']].agg('sum').reset_index().rename(index=str,columns={'click_times':'last2click'})
    f1 = log[['USRID']].drop_duplicates()
    f1= pd.merge(f1,f,on='USRID',how='left')
    f1=f1.fillna(0)
    data =  pd.merge(data,f1,on='USRID',how='left')
    data['last2dayclick/mean'] = data.last2click/data.mean_day_click
    data['last2dayclick/total'] = data.last2click/data.EVT_LBL_len
    del data['last2click']
    
    return data


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

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



#最后2，7天的预测值作为特征
def prediction_feature(log,data):
    log_last7day = log[log.week_number==13]
    data_last7day = basicfeature(log_last7day,data_raw)
    va,tt,fp=kfold_lightgbm(data_last7day,5,True)
    rst = pd.concat([va[['USRID','RST']],tt[['USRID','RST']]],axis=0)
    rst1 = rst.rename(index=str,columns={'RST':'last7day_RST'})
    
    log_last2day = log[log.day>=30]
    data_last2day = basicfeature(log_last2day,data_raw)
    va,tt,fp=kfold_lightgbm(data_last2day,5,True)
    rst = pd.concat([va[['USRID','RST']],tt[['USRID','RST']]],axis=0)
    rst2 = rst.rename(index=str,columns={'RST':'last2day_RST'})
    return rst1,rst2

# Drop高相关性特征 
def corrtest(data):
    corrs = data.corr()
    threshold = 0.85
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



def kfold_lightgbm(df, num_folds, stratified = False):
    # Divide in training/validation and test data
    train_df = df[df['FLAG']!=-1]
    test_df = df[df['FLAG']==-1]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=False, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['FLAG','USRID']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['FLAG'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['FLAG'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['FLAG'].iloc[valid_idx]
        
          
        clf = lgb.LGBMClassifier(
        nthread=4,
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        colsample_bytree=0.997212866002197,
        bagging_fraction=0.7733927534732657,
        min_data_in_leaf=37,
        min_child_weight = 13.05659547343758,
        min_split_gain=0.027258234021548238,
        reg_lambda=0.12367585365238067,
        verbose=0)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 150)

        oof_preds[valid_idx]  = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y,  oof_preds[valid_idx] )))
        
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], oof_preds))
    # Write submission file and plot feature importance
    train_df['RST'] = oof_preds
    test_df['RST'] = sub_preds
    test_df[['USRID', 'RST']].to_csv('submission6.csv', index= False,sep='\t')
    display_importances(feature_importance_df)
    return train_df[['USRID', 'RST']],test_df[['USRID', 'RST']],feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')




def kfold_xgb(df, num_folds, stratified = False):
    # Divide in training/validation and test data
    train_df = df[df['FLAG']!=-1]
    test_df = df[df['FLAG']==-1]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=False, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['FLAG','USRID']]
    feature_importance_df=pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['FLAG'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['FLAG'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['FLAG'].iloc[valid_idx]
        
        train_x = xgb.DMatrix(train_x,label=train_y)
        valid_x = xgb.DMatrix(valid_x,label=valid_y)
        
        params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'max_depth':4,
	    'subsample':0.85,
	    'colsample_bytree':0.8,
	    'colsample_bylevel':0.8,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':4,
        'gamma':0.5,
        'min_child_weight':50,
	    }
       
        watchlist = [(train_x,'train'),(valid_x,'val')]
        clf = xgb.train(params,train_x,num_boost_round=3000,evals=watchlist,early_stopping_rounds=90)

        test = xgb.DMatrix(test_df[feats])
        oof_preds[valid_idx] = clf.predict(valid_x,ntree_limit=clf.best_ntree_limit)
        sub_preds += clf.predict(test, ntree_limit=clf.best_ntree_limit) / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        xgb.plot_importance(clf)
        fscore = clf.get_fscore()   
        a=list(fscore.keys())
        v=list(fscore.values())
        fold_importance_df=pd.DataFrame()
        fold_importance_df['feature'] = a
        fold_importance_df['importance'] = v
        fold_importance_df['fold'] = n_fold+1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
         
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], oof_preds))
    # Write submission file and plot feature importance
    train_df['RST'] = oof_preds
    test_df['RST'] = sub_preds
   # test_df[['USRID', 'RST']].to_csv('submission6.csv', index= False,sep='\t')
    return train_df[['USRID', 'RST']],test_df[['USRID', 'RST']],feature_importance_df


if __name__ == "__main__":
    data = basicfeature(log,data_raw)
    print('Basic feature done')
   
    data = extrafeature(log,data)
    print('Extra feature done')
   
    level1_distribution = clickdistribution(log)
    data =  pd.merge(data,level1_distribution,on='USRID',how='left')
    print('Level1 distribution done')
   
    corr= corrtest(data)
    data=data.drop(corr,axis=1)
    print('Drop high corr feature done')
    
    rst1,rst2 = prediction_feature(log,data_raw)   
    data =  pd.merge(data,rst1,on='USRID',how='left')
    data =  pd.merge(data,rst2,on='USRID',how='left')
    print('Prediciton feature done')
    
    strfea=['V2','V3','V4','V5']
    data[strfea] = data[strfea].astype(str)
    data,col_new = one_hot_encoder(data)
    
    data.to_csv('stackingdata1.csv',index=False)







#856620



