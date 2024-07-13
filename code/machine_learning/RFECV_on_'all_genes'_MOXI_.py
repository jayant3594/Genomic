
from google.colab import files
'''
The codes were run on 'Google colab'
/*Change the input data code; to run it on Jupyter*/
'''
uploaded = files.upload()

# Commented out IPython magic to ensure Python compatibility.
#File Importing and Setting Data 
import numpy as np
import pandas as pd
import io
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
# %matplotlib inline

#.....................................................................................#

df = pd.read_csv(io.BytesIO(uploaded['all_gene_mutation.csv']),low_memory=False)
df.head()

# Removing correlated features

correlated_features = set()
correlation_matrix = df.drop(columns=['Isolate_no','Class'], axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

len(correlated_features)

df_no_corr = df.drop(columns=list(correlated_features), axis=1)
print("Shape of 'df_no_corr' = ", df_no_corr.shape)
df_no_corr.head()
#.....................................................................................#

class Data():
    '''
    Created a Class object for initial Data processing
    '''
    def __init__(self):
        pass
    
    def get_data(self,csv_path):
        '''
        Input the data file path as 'csv_path'
        Returns output as Pandas Dataframe
        '''
        return pd.read_csv(csv_path,low_memory=False)
    
    def feature_scaling(self,features):
        '''
        Input a feature data in DataFrame
        Returns Normalized features
        '''
        m,n = features.shape
        for i in range(n):
            if features.iloc[:,i].min() != features.iloc[:,i].max():
                features.iloc[:,i] = (features.iloc[:,i]-features.iloc[:,i].min())/(
                    (features.iloc[:,i].max()-features.iloc[:,i].min()))
        return features
    
    def get_features(self,df,scaling=False, column_name=[]):
        '''
        'df' must be in Pandas Dataframe
        scaling is by-default false for an attribute
        Insert column_name which may be droped:  e.g. column_name=['isolate_no','class']
        '''
        features = df
        for col in column_name:
            features = features.drop(col,axis=1) #Drop the columns
        if scaling == True:
            features = Data.feature_scaling(self,features)
        features = np.c_[features]
        return features
    
    def get_targets(self,df,scaling=False, column_name=[]):
        '''
        scaling is by-default false for an attribute
        Insert column_name which will become a target matrix:  e.g. column_name=['class']
        '''
        targets = df
        for col in column_name:
            targets = targets[col]
        if scaling == True:
            targets = Data.feature_scaling(self,targets)
        return targets

#.....................................................................................#

data = Data()

X = data.get_features(df_no_corr,scaling=True, column_name=['Isolate_no','Class'])
print("Shape of feature matrix : ",X.shape)
y = data.get_targets(df_no_corr, column_name=['Class'])
print("Shape of target matrix : ",y.shape)

#.....................................................................................#

# RFECV on un-correlated feature Gene matrix

# Defining classifier
classifier = AdaBoostClassifier(n_estimators=50,learning_rate=0.75)

rfe=RFECV(classifier,1,cv=StratifiedKFold(10),n_jobs=-1,scoring='roc_auc',verbose=1)
fit=rfe.fit(X,y)
print("Optimal number of features : %d" % fit.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(fit.grid_scores_) + 1), fit.grid_scores_)
plt.show()

print('Optimal number of features: {}'.format(fit.n_features_))
len(rank)

#.....................................................................................#

import copy

df_no_corr_copy1 = df_no_corr.copy()
df_no_corr_copy1

df_no_corr_copy1 = df_no_corr_copy1.drop(columns= ['Isolate_no', 'Class'], axis=1)
df_no_corr_copy1

print(np.where(fit.support_ == False)[0])

df_no_corr_copy1.drop(df_no_corr_copy1.columns[np.where(fit.support_ == False)[0]], axis=1, inplace=True)
df_no_corr_copy1

dset = pd.DataFrame()
dset['attr'] = df_no_corr_copy1.columns
dset['importance'] = fit.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)

# Plotting figure of 'Feature v/s important score'
plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

#.....................................................................................#

dset.to_csv('dset.csv')
files.download('dset.csv')

df_no_corr_copy2 = df_no_corr.copy()
df_no_corr_copy2.head()

df_no_corr_copy2 = df_no_corr_copy2.drop(columns= ['Isolate_no', 'Class'], axis=1)
df_no_corr_copy2.head()
df_no_corr_copy2.shape

rank=fit.ranking_
imp=pd.DataFrame(list(zip(df_no_corr_copy2.columns.tolist(),rank)),columns=['Feature','Rank'])
imp=imp.sort_values('Rank',ascending=True)

imp.to_csv('fit_all_Rank.csv')
files.download('fit_all_Rank.csv')

#------------------------------------------******--------------------------------------------#
