
import pandas as pd
import numpy as np
'''
This code finds out the gene/ intergenic regions,
which show most number of total mutation and arrange them in descending order
'''

gene = pd.read_csv("...\\PROJECT\\mutation_data\\all_gene_mutation.csv")
gene.head()
gene_arr = np.array(gene)[:,1:-1].astype('int')
gene1 = gene.drop(['Isolate_no','Class'], axis = 1) 


dict1 = {'feature':[], 
        'value_sum':[]} 
df2 = pd.DataFrame(dict1) 
for ind,j in enumerate(gene1):
    df2.loc[len(df2.index)] = [j, gene_arr[:,ind].sum()/240]   # [j, gene1[j].sum()/120] 
    
print(df2.head())
df2.sort_values(by=['value_sum'], inplace=True,ascending=False)
df2.head(10)

########################################################################################################

gyr = pd.read_csv("...\\PROJECT\\mutation_data\\gyr_gene_mutation.csv")
gyr.head()
gyr1 = gyr.drop(['Isolate_no','Class'], axis = 1) 
gyr_arr = np.array(gyr1).astype('int')

dict2 = {'feature':[], 
        'value_sum':[]} 
df3 = pd.DataFrame(dict2) 
for ind,j in enumerate(gyr1):
    df3.loc[len(df3.index)] = [j, gyr_arr[:,ind].sum()/240]   # [j, gene1[j].sum()/120] 
    
print(df3.head())
df3.sort_values(by=['value_sum'], inplace=True,ascending=False)
df3.head(10)

########################################################################################################

interg = pd.read_csv("...\\PROJECT\\mutation_data\\intergenic_mutation.csv")
interg.head()

interg1 = interg.drop(['Isolate_no','Class'], axis = 1) 
interg_arr = np.array(interg1).astype('int')

dict3 = {'feature':[], 
        'value_sum':[]} 
df4 = pd.DataFrame(dict3) 
for ind,j in enumerate(interg1):
    df4.loc[len(df4.index)] = [j, interg_arr[:,ind].sum()/240] 
    
print(df4.head())
df4.sort_values(by=['value_sum'], inplace=True,ascending=False)
df4.head(10)

########################################################################################################
