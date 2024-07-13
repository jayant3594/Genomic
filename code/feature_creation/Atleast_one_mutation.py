
import pandas as pd
import numpy as np

ref = pd.read_excel("...\\MTb_Genes.xlsx")
ref.head(3)

gene = pd.read_csv("...\\mutation_data\\all_gene_mutation.csv")
gene.head()
gene_arr = np.array(gene)[:,1:-1].astype('int')
gene1 = gene.drop(['Isolate_no','Class'], axis = 1) 


dict1 = {'feature':[], 
        'value_sum':[]} 
df2 = pd.DataFrame(dict1) 
for ind,j in enumerate(gene1):
    df2.loc[len(df2.index)] = [j, gene_arr[:,ind].sum()/240]   # [j, gene1[j].sum()/120] 
    
print(df2.head())
df2.head(10)

atleast_one_mutation = []
no_mutation = []
a = df2['value_sum']
b = df2['feature']
for i,j in enumerate(a):
    if j == 0:
        no_mutation.append(b[i])
    elif j != 0: 
        atleast_one_mutation.append(b[i])
print('atleast one mutation genes = ',len(atleast_one_mutation))
print('no mutation genes = ',len(no_mutation))
print('gene data length = ',gene.shape)

dict2 = {'GeneID':[],
         'Locus':[],
        'Locus tag':[],
        'Protein Name':[]}

dict3 = {'GeneID':[],
         'Locus':[],
        'Locus tag':[],
        'Protein Name':[]}

atleast_one_mutation_genes = pd.DataFrame(dict2)
no_mutation_genes = pd.DataFrame(dict3)

ref_gene = np.array(ref)
geneid = ref_gene[:,5]

for ind,j in enumerate(geneid):
    if str(j) in atleast_one_mutation:
        atleast_one_mutation_genes.loc[len(atleast_one_mutation_genes.index)] = [ref_gene[ind,5], 
                                                                                 ref_gene[ind,6], 
                                                                                 ref_gene[ind,7], ref_gene[ind,10]]
    elif str(j) in no_mutation:
        no_mutation_genes.loc[len(no_mutation_genes.index)] = [ref_gene[ind,5], 
                                                               ref_gene[ind,6],
                                                               ref_gene[ind,7], ref_gene[ind,10]]


print(atleast_one_mutation_genes.head())
print(no_mutation_genes.head())


atleast_one_mutation_genes.to_excel((r'...\\Genes with-without mutation\\atleast_one_mutation_genes.xlsx'), index=False)
no_mutation_genes.to_excel((r'...\\Genes with-without mutation\\no_mutation_genes.xlsx'), index=False)

##################################################################################################################################################################