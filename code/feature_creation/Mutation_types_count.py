
import pandas as pd
import numpy as np
'''
This code separates the mutation type and 
creates a new excel/csv file based on type of mutaion for a isolate
'''

# Uploading .csv file contains all the SRR accession no of resistant and suseptible isolates
srr = pd.read_csv("...\\PROJECT\\SRR list\\pvt.csv")
srr.head(3)

# creating lists of SRR accession no for resistant and suseptible isolates
res_list  = np.array(srr.iloc[:,1])
sus_list = np.array(srr.iloc[:,2])

data = np.array(['isolate_no','snp','ins','del','complex','mnp','Total'])
my_data = pd.DataFrame(data=data).T


####################################################################################################

clas = 0    # Enter the class => 1: resistant, 0: suseptible

for name in sus_list:                  # Taking every SRR no in res_list which represents suseptible isolate data
    isolate_no = name+'_'+str(clas)    # Saving the isolate sample no. using name and class
    
    # Upload the .csv file; converted from "VCF"
    df = pd.read_csv("...\\PROJECT\\Suseptible\\combine_csv_sus\\"+name+".csv")
    TYPE = np.array(df.iloc[:,7])
    TYPE = list(TYPE)
    
    my_zeros = np.zeros(7)
    my_zeros[1] = str(TYPE).count('TYPE=snp')
    my_zeros[2] = str(TYPE).count('TYPE=ins')
    my_zeros[3] = str(TYPE).count('TYPE=del')
    my_zeros[4] = str(TYPE).count('TYPE=complex')
    my_zeros[5] = str(TYPE).count('TYPE=mnp')
    my_zeros[6] = str(TYPE).count('TYPE')
    
    df_count = pd.DataFrame(data=my_zeros)
    df_count.iloc[0,0] = isolate_no
    
    my_data = pd.concat([my_data, df_count.T], ignore_index = True)


my_data.to_csv((r'...\\res_count.csv'), index=False, header=False)

##############################################################################################################