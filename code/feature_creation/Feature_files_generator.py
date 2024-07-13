
import numpy as np
import pandas as pd
'''
This single code takes 6 .csv files initially
Then it accepts the generated VCF from 'usegalaxy.eu' in converted .csv format (total 240 files)
Using the mutations in all 240 files
It generates three types of feature files in a single run
    first : Mutation in all gene (named as 'all_gene_mutation.csv')
    Second : Mutation in all Intergenic region (named as 'intergenic_mutation.csv')
    Third : Mutation in every position of gyrB and gyrA genes (named as 'gyr_gene_mutation.csv')
'''
import copy


# Uploading gene Position data of MTB
ref = pd.read_excel("...\\PROJECT\\MTb_Genes.xlsx") 
ref.head(3)

# Uploading Intergenic region position data
intergenic_region = pd.read_csv("...\\PROJECT\\Intergenic_region.csv")
intergenic_region.head(3)

# Uploading .csv file contains all gene name in each column
gene_column = np.array(ref['GeneID'])
iso_id = np.array(['Isolate_no'])
clas = np.array(["Class"])
mutation = np.hstack((iso_id,gene_column,clas))
mutation = pd.DataFrame(data = [],columns = mutation)
myvcf_mutation = mutation.copy()
mutation.head()


# Uploading .csv file contains gyrB and gyrA gene's all position in different columns
gyr_mutation = pd.read_csv("...\\PROJECT\\gyr_mutation.csv")

gyr_mutation = gyr_mutation.drop(list(range(120)))       # Droping all the rows of zero
gyr_mutation.insert(gyr_mutation.shape[1],"Class",[])    # inserting column name 'Class' at the last
myvcf_gyr = gyr_mutation.copy()
gyr_mutation.head()


# Uploading .csv file contains intergenic region name in each column

intrg_column = np.array(intergenic_region['intergenic_region'])
intergenic_mutation = np.hstack((iso_id,intrg_column,clas))
intergenic_mutation = pd.DataFrame(data = [],columns = intergenic_mutation)
myvcf_inter = intergenic_mutation.copy()
intergenic_mutation.head()

########################################################################################################

ref_num = np.array(ref)    # converting dataframe into array
m,n = ref_num.shape        # abstacting length in both direction
p = intergenic_region.shape[0]   # Length of intergenic region

# Creating a list which includes all the gene position of 'gyrB' and 'gyrA'
gyr_list = list(range(5240,7268)) + list(range(7302,9819))       
inter_regnum = np.array(intergenic_region)   # converting dataframe into array

########################################################################################################

#************* Code for data processing ****************#

def save_mutation(file, name, add_to, isolate_no, clas=1,
                 mutate = False, gyr = False, interg = False):
    '''
    'file' should be mutation in zeros array (e.g. mut_zero)
    'name' should be what you want to save as a .csv (e.g. SRR1234567)
    'add_to' is original dataframe (e.g. mutation) where all isolate mutations are going to stack-up 
    '''
    file[-1] = clas               # Updating 'class' in an array
    df = pd.DataFrame(file)       # converting numpy file into Dataframe
    df.iloc[0,0] = isolate_no     # Updating 'isolate no' in DataFrame
    
    name = str(name)+'_'+str(clas)+'.csv'         # Updating given name( e.g. SRR123457_1.csv)
    df = df.T                                     # Transforming dataframe
    df.columns = add_to.columns                   # Changing column names of 'df' into columns of 'add_to'
    
    if mutate == True:
        # downloading dataframe as a csv format in this directory only if condition is 'True'
        df.T.to_csv((r'...\\PROJECT\\mutation_data\\mutate\\'+ name), header=False) 
    elif gyr == True:
        df.T.to_csv((r'...\\PROJECT\\mutation_data\\gyr_mutate\\'+ name), header=False)  
    elif interg == True:
        df.T.to_csv((r'...\\PROJECT\\mutation_data\\interg_mutate\\'+ name), header=False)
    else:
        # If none of the above conditions are true, then it will be saved in this directory
        df.T.to_csv((r'...\\PROJECT\\mutation_data\\'+ name), header=False)
        
    add_to =  pd.concat([add_to, df], ignore_index = True)    # Returning concated dataframe
    return add_to

########################################################################################################

# Uploading .csv file contains all the SRR accession no of resistant and suseptible isolates
srr = pd.read_csv("...\\Project\\SRR_list\\pvt.csv")
srr.head(3)

# creating lists of SRR accession no for resistant and suseptible isolates
res_list  = np.array(srr.iloc[:,1])
sus_list = np.array(srr.iloc[:,2])

########################################################################################################

#****************************     Loop Starts from here    *************************#

clas = 0    # Enter the class => 1: resistant, 0: suseptible

for name in sus_list:                  # Taking every SRR no in sus_list which represents suseptible isolate data
    isolate_no = name+'_'+str(clas)    # Saving the isolate sample no. using name and class
    # Upload the .csv file; converted from "VCF"
    df = pd.read_csv("...\\PROJECT\\Susceptible\\combine_csv_sus\\"+name+".csv")
    
    # creating three array of zeros in respective length
    mut_zero = np.zeros(mutation.shape[1])
    gyrmut_zero = np.zeros(gyr_mutation.shape[1])
    intermut_zero = np.zeros(intergenic_mutation.shape[1])

    mutation_position = np.array(df)[:,1]   # creating array of postion's from file 'df' 

    for j in range(m):                      # range is length of m: (length of 'ref' dataframe)

        # for a gene region
        gene_start = ref_num[j,2]           # taking staring position of NCBI Gene range
        gene_stop = ref_num[j,3]            # taking stoping position of NCBI Gene range
        gene_id = ref_num[j,5]              # taking id of NCBI Gene in above mentioned range 

        # for an intergenic region
        if j < p:
            inter_start = inter_regnum[j,0]     # taking staring position of intergenic region
            inter_stop = inter_regnum[j,1]      # taking stoping position of intergenic region
            inter_id = inter_regnum[j,2]        # taking id of intergenic region in above mentioned range 

        '''
        Now, here; updating three array of zeros which was defined earlier
        After updation, they will consist the mutation 
        for 'every gene-range', 'gyr gene-range' and 'intergenic-range' respectively
        '''

        for position in mutation_position:      # for every position in mutation_position array
            
            if (position >= gene_start) and (position <= gene_stop): 
                '''
                this code adds every mutation if it falls under the NCBI gene range
                '''
                mut_zero[j+1] += 1      # if a position caught in a range of particular gene, it will add 1 at that gene
                # Here (j+1) is used because first column is 'isolate_no'
                '''
                this below code adds every mutation if it falls under the gyrB(887081) and gyrA(887105) gene range
                '''
                if gene_id == 887081:
                    for index,num in enumerate(gyr_list[0:2028]):   # range of gyrB in list gyr_list
                        if num == position:
                            gyrmut_zero[index+1] += 1  # 1 is added to 'index'; because the first column is 'isolate_no'
                            break
                elif gene_id == 887105:
                    for index,num in enumerate(gyr_list[2028:]):     # range of gyrA in list gyr_list
                        if num == position:
                            gyrmut_zero[index+2029] += 1             # gyrA starts from(index+1+2028)
                            break

            elif (j < p) and (position >= inter_start) and (position <= inter_stop):
                '''
                this code adds every mutation if it falls under intergenic region
                '''
                intermut_zero[j+1] += 1
                # Here (j+1) is used because first column is 'isolate_no'
                
    myvcf_mutation = save_mutation(mut_zero,name, myvcf_mutation, isolate_no, clas, mutate = True)
    myvcf_gyr = save_mutation(gyrmut_zero, name, myvcf_gyr, isolate_no, clas, gyr = True)
    myvcf_inter = save_mutation(intermut_zero, name, myvcf_inter, isolate_no, clas, interg = True)


# *************************   Saving Data files   ************************* #

myvcf_mutation.to_csv((r'...\\PROJECT\\mutation_data\\all_gene_mutation.csv'), index=False)
myvcf_gyr.to_csv((r'...\\PROJECT\\mutation_data\\gyr_gene_mutation.csv'), index=False)
myvcf_inter.to_csv((r'...\\PROJECT\\mutation_data\\intergenic_mutation.csv'), index=False)

##################################################################################################################
