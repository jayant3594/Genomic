
import numpy as np
import pandas as pd
import copy


# Uploading gene Position data of MTB
ref = pd.read_excel("...\\MTb_Genes.xlsx")
ref.head(3)

# Uploading Intergenic region position data
intergenic_region = pd.read_csv("...\\Intergenic_region.csv")
intergenic_region.head(3)

########################################################################################################
# creating arrays

# for geneomic region
gene_start = np.array(ref['Start'])            # taking staring positions of Gene range
gene_stop = np.array(ref['Stop'])              # taking stoping positions of Gene range
gene_Locus_tag = np.array(ref['Locus tag'])    # taking Locus tag from 'ref' data
gene_protein = np.array(ref['Protein Name'])   # taking Protein from 'ref' data

# for intergenic region
inter_start = np.array(intergenic_region['start'])     # taking staring positions of intergenic region
inter_stop = np.array(intergenic_region['stop'])       # taking stoping positions of intergenic region
inter_id = np.array(intergenic_region['intergenic_region'])  # taking id of intergenic region in above mentioned range

########################################################################################################

# Uploading .csv file contains all the SRR accession no of resistant and suseptible isolates
srr = pd.read_csv("...\\Moxifloxacin\\SRR_list\\pvt.csv")
srr.head(3)

# creating lists of SRR accession no for resistant and suseptible isolates
res_list  = np.array(srr.iloc[:,1])
sus_list = np.array(srr.iloc[:,2])

########################################################################################################

# clas = 1    # Enter the class => 1: resistant
clas = 0    # Enter the class => 0: suseptible

for name in sus_list:                  # Taking every SRR no in sus_list which represents suseptible isolate data
    isolate_no = name+'_'+str(clas)    # Saving the isolate sample no. using name and class
    # Upload the SNPs data for an isolate as a .csv file; converted from "VCF"
    df = pd.read_csv("...\\Moxifloxacin\\Susceptible\\combine_csv_sus\\"+name+".csv")
    
    df_Locus_tag = pd.DataFrame(data=np.ones(df.shape[0]))     # creating a dataframe as length as snp file
    df_Protein_Name = pd.DataFrame(data=np.ones(df.shape[0]))
    position = np.array(df['POS'])     # creating an array of all 'POS' from snps data

    for ind,pos in enumerate(position):
        '''
        This code finds out if a position falls under genomic region or intergenic region
            INPUT: 
                'POS' data
                'Gene' and 'Intergenic region' range
                'Locus tag', 'Protein name' and 'Intergenic region name'
            OUTPUT:
                Dataframe1: 'Locus tag' that will be used in next to position 
                Dataframe2: 'Protein name' that will be used in next to position
        '''
        for num in range(len(gene_start)):

            if (pos >= gene_start[num]) and (pos <= gene_stop[num]):

                # 'POS' is falling under genomic region
                df_Locus_tag.iloc[ind] = gene_Locus_tag[num]
                df_Protein_Name.iloc[ind] = gene_protein[num]
                break

            elif (num < len(inter_start)) and (pos >= inter_start[num]) and (pos <= inter_stop[num]):

                # 'POS' is falling under Intergenic region
                df_Locus_tag.iloc[ind] = inter_id[num]
                df_Protein_Name.iloc[ind] = '-'
                break
                
    df2 = df.copy()
    df2['Locus tag'] = df_Locus_tag        # Adding first ouput from 'main code' into dataframe
    df2['Protein Name'] = df_Protein_Name  # Adding second ouput from 'main code' into dataframe
    
    # creating dataframe in required format
    df3 = pd.concat([df2[df2.columns[0:5]], df2[df2.columns[-2:]],df2[df2.columns[5:-2]]],axis=1)
    
    df3.to_csv((r"...\\Moxifloxacin\\Susceptible\\Rv_number_sus\\"+isolate_no+".csv"),index = False)

####################################################################################################################################