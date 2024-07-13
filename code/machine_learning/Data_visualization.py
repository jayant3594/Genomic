
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

csv_path = "C:\\...\\all_gene_mutation.csv"
df = pd.read_csv(csv_path,low_memory=False)
df.head()

#####################################  Box-Plots  #####################################

# Checking for top 25 'highest median' features
print(df.median().sort_values(ascending=False)[0:25]) 

# making a group of features with highest median
k = df.median().sort_values(ascending=False)[0:25].index

# Visualization (Box-plots) of top 25 features with highest median value
plt.figure(figsize=(15,8),dpi=600)
plt.xticks(rotation=30,fontsize=14)
sns.set(color_codes = 'lightgray')

my_plt = sns.boxplot(data=df[list(k)], medianprops=dict(color="red", alpha=0.8))
plt.xlabel('Gene ID',size=18)
plt.ylabel('Mutation count',size=18)
plt.show()
#------------------------------------------------------------------------#

# Box-plots for four genes (features) related with 2nd line drugs target
plt.figure(figsize=(6,3),dpi=600)
plt.xticks(rotation=30,fontsize=9)
sns.set(color_codes = 'lightgray')

my_plt_four_genes = sns.boxplot(data=df[['887081','887105','885903','885396']], medianprops=dict(color="red", alpha=0.8))
plt.xlabel('Gene ID',size=11)
plt.ylabel('Mutation count',size=11)
plt.show()
#------------------------------------------------------------------------#

# Seperating Gyrase A/B data into Resistant & Susceptible isolates

gene_887081_res = df['887081'][0:120]
gene_887081_sus = df['887081'][120:]
gene_887081_res.reset_index(drop = True, inplace = True)
gene_887081_sus.reset_index(drop = True, inplace = True)

gene_887105_res = df['887105'][0:120]
gene_887105_sus = df['887105'][120:]
gene_887105_res.reset_index(drop = True, inplace = True)
gene_887105_sus.reset_index(drop = True, inplace = True)

# Making a new dataframe
gyrase = pd.DataFrame({'gene_887081_res':gene_887081_res,
                     'gene_887081_sus':gene_887081_sus,
                    'gene_887105_res':gene_887105_res,
                    'gene_887105_sus':gene_887105_sus})
print(gyrase)
#------------------------------------------------------------------------#

# Box-plots for Gyarse A/B genes (features) seperated into Resistant/Susceptible

plt.figure(figsize=(6,3),dpi=600)
plt.xticks(rotation=30,fontsize=9)
sns.set(color_codes = 'lightgray')

my_plt_gyrase = sns.boxplot(data=gyrase, color = 'b', medianprops=dict(color="red", alpha=0.8))
plt.xlabel('Gene ID',size=11)
plt.ylabel('Mutation count',size=11)
plt.show()
#------------------------------------------------------------------------#

# Data inspect
print(gene_887081_res.median())
print(gene_887081_res.mean())
print("")
print(gene_887081_sus.median())
print(gene_887081_sus.mean())
print("")
print(gene_887105_res.median())
print(gene_887105_res.mean())
print("")
print(gene_887105_sus.median())
print(gene_887105_sus.mean())

#####################################  Correlations  #####################################

df_new = df.copy() # Making copy of origional data
corr_matrix = df_new.corr()  # Finding correlations among features & target

# Checking correaltion of 'Class' with other features (genes)
corr_matrix['Class'].sort_values(ascending=False)[0:10]

# Scatter plot b/w 'Class' and '887132' (top +ve correlated feature)
plt.figure(figsize=(5,3),dpi=100)
sns.scatterplot(x='Class', y='887132', data=df_new, color='r') #hue='Class')
plt.show()
#------------------------------------------------------------------------#

corr_matrix['Class'].sort_values(ascending=True)[0:10]  # Checking negative correlations

# Scatter plot b/w 'Class' and '888515' (top -ve correlated feature)
plt.figure(figsize=(5,3),dpi=100)
sns.scatterplot(x='Class', y='888515', data=df_new, color='g')
plt.show()
#------------------------------------------------------------------------#

# Scatter plot b/w 'Class' and '887105' (GyrA)
plt.figure(figsize=(5,3),dpi=100)
sns.scatterplot(x='Class', y='887105', data=df_new, color='k')
plt.show()
#------------------------------------------------------------------------#

# Scatter plot b/w 'Class' and '887081' (GyrB)
plt.figure(figsize=(5,3),dpi=100)
sns.scatterplot(x='Class', y='887081', data=df_new, color='darkviolet')
plt.show()
#------------------------------------------------------------------------#

# creating dataframe of some ideal random data
random_data = pd.DataFrame({'Feature':[8,9,5.6,6.4,9.9,7.2,1,4,3.5,2.6,1.2,0.3], 'Class':[1,1,1,1,1,1,0,0,0,0,0,0]})

# Scatter plot b/w 'Class' and 'random_data'
plt.figure(figsize=(7,3),dpi=100)
sns.lmplot(x='Class', y='Feature', data=random_data, height=5)#, color='darkviolet')
plt.show()

#-----------------------------------------   ***   -----------------------------------------#