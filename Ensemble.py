# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:39:45 2018

@author: Allen
"""



import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import rankdata
import gc
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV

data1 = pd.read_csv('stackingdata1.csv')
data2 = pd.read_csv('stackingdata2.csv')

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

va1,op1,fp1=kfold_lightgbm(data2,5,True)
va2,op2,fp2=kfold_xgb(data2,5,True)

va3,op3,fp3=kfold_lightgbm(data1,5,True)
va4,op4,fp4=kfold_xgb(data1,5,True)


#############


va1=pd.DataFrame(va1)
va2=pd.DataFrame(va2)

va3=pd.DataFrame(va3)
va4=pd.DataFrame(va4)

rank_d1_lgb = rankdata(va1.RST, method='ordinal')
rank_d1_xgb = rankdata(va2.RST, method='ordinal')

finalRank_dataset1 = (rank_d1_lgb+rank_d1_xgb*0.8)
finalRank_dataset1=finalRank_dataset1/(max(finalRank_dataset1) + 1.0)

train_df = data1[data1['FLAG']==-1]
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], va1['RST']))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], va2['RST']))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], finalRank_dataset1))


rank_d2_lgb = rankdata(va3.RST, method='ordinal')
rank_d2_xgb = rankdata(va4.RST, method='ordinal')

finalRank_dataset2 = (rank_d2_lgb*0.8+rank_d2_xgb)
finalRank_dataset2=finalRank_dataset2/(max(finalRank_dataset2) + 1.0)
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], va3['RST']))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], va4['RST']))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], finalRank_dataset2))


finalRank_dataset1 = (rank_d1_lgb+rank_d1_xgb*0.8)
finalRank_dataset2 = (rank_d2_lgb*0.8+rank_d2_xgb)
finalRank_dataset = (finalRank_dataset1+finalRank_dataset2*0.7)
finalRank_dataset=finalRank_dataset/(max(finalRank_dataset) + 1.0)
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], finalRank_dataset1))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], finalRank_dataset2))
print('Full AUC score %.6f' % roc_auc_score(train_df['FLAG'], finalRank_dataset))



#output
op1=pd.DataFrame(op1)
op2=pd.DataFrame(op2)

op3=pd.DataFrame(op3)
op4=pd.DataFrame(op4)

rank_d1_lgb = rankdata(op1.RST, method='ordinal')
rank_d1_xgb = rankdata(op2.RST, method='ordinal')
rank_d2_lgb = rankdata(op3.RST, method='ordinal')
rank_d2_xgb = rankdata(op4.RST, method='ordinal')

finalRank_dataset1 = (rank_d1_lgb+rank_d1_xgb*0.8)
finalRank_dataset2 = (rank_d2_lgb*0.8+rank_d2_xgb)
finalRank_dataset = (finalRank_dataset1+finalRank_dataset2*0.7)


test_df = data1[data1['FLAG']==-1]
test_df['RST'] = finalRank_dataset
test_df[['USRID', 'RST']].to_csv('submission0714.csv', index= False,sep='\t')






