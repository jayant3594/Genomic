
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#...................................................................................#

# Chi2 on Moxifloxacin data

csv_path = "C:\\...\\all_gene_mutation.csv"
df = pd.read_csv(csv_path,low_memory=False)
df.head()

# Defining X and y
X = df.iloc[:,1:-1]
y = df.iloc[:,-1:]

#apply SelectKBest class to extract top 'all' best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(50,'Score'))  #print 50 best features

featureScores2 = featureScores.nlargest(3906,'Score')

# Downloading feature Ranking
featureScores2.to_csv((r"C:\\...\\chi2_all_features_all_genes.csv"), index = False)

#...................................................................................#

# Chi2 on Ofloxacin data

csv_path1 = "C:\\...\\gene.csv"
df_ofl = pd.read_csv(csv_path1,low_memory=False)
df_ofl.head()

# Defining X and y
X_ofl = df_ofl.iloc[:,1:-1]
y_ofl = df_ofl.iloc[:,-1:]

#apply SelectKBest class to extract top 'all' best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit_ofl = bestfeatures.fit(X_ofl,y_ofl)
dfscores_ofl = pd.DataFrame(fit_ofl.scores_)
dfcolumns_ofl = pd.DataFrame(X_ofl.columns)
#concat two dataframes for better visualization 
featureScores_ofl = pd.concat([dfcolumns_ofl,dfscores_ofl],axis=1)
featureScores_ofl.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores_ofl.nlargest(50,'Score'))  #print 50 best features

featureScores_ofl2 = featureScores_ofl.nlargest(3852,'Score')

# Downloading feature Ranking
featureScores_ofl2.to_csv((r"C:\\...\\chi2_all_features_all_genes_ofl.csv"), index = False)

#----------------------------------------***------------------------------------------------#