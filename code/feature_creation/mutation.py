
import numpy as np
import pandas as pd

# This code was created using Jupyter Notebook
# importing reference gene data
ref = pd.read_csv("...\\NCBI_gene.csv")
ref.head(3)

# importing intergenic region data
intergenic_region = pd.read_csv("...\\Intergenic_region.csv")
intergenic_region.head(3) 


# importing genome data
mutation = pd.read_csv("...\\genome.csv")
mutation = mutation.drop(list(range(120)))     # Droping all the rows of zero
mutation.insert(mutation.shape[1],"class",[])  # inserting column name 'class' at the last
myvcf_mutation = mutation
mutation.head()

# importing gyr data
gyr_mutation = pd.read_csv("...\\gyr_mutation.csv")
gyr_mutation = gyr_mutation.drop(list(range(120)))       # Droping all the rows of zero
gyr_mutation.insert(gyr_mutation.shape[1],"class",[])    # inserting column name 'class' at the last
myvcf_gyr = gyr_mutation
gyr_mutation.head()

# importing intergenic data
intergenic_mutation = pd.read_csv("...\\Intergenic_mutation.csv")
intergenic_mutation = intergenic_mutation.drop(list(range(120)))      # Droping all the rows of zero
intergenic_mutation.insert(intergenic_mutation.shape[1],"class",[])   # inserting column name 'class' at the last
myvcf_inter = intergenic_mutation
intergenic_mutation.head()

#########################################################################################

ref_num = np.array(ref)    # converting dataframe into array
m,n = ref_num.shape        # abstacting length in both direction

# Creating a list which includes all the gene position of 'gyrB' and 'gyrA'
gyr_list = list(range(5240,7268)) + list(range(7302,9819))       
inter_regnum = np.array(intergenic_region)   # converting dataframe into array

########################################################################################

#*************                                  ************#
#$$$$$$$$$$$ *** Code for data processing *** $$$$$$$$$$#
#*************                                  ************#

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
        df.T.to_csv((r'C:...\\mutate\\'+ name), header=False) 
    elif gyr == True:
        df.T.to_csv((r'C:...\\gyr_mutate\\'+ name), header=False)  
    elif interg == True:
        df.T.to_csv((r'C:...\\interg_mutate\\'+ name), header=False)
    else:
        # If none of the above conditions are true, then it will be saved in this directory
        df.T.to_csv((r'C:...\\mutation_data\\'+ name), header=False)
        
    add_to =  pd.concat([add_to, df], ignore_index = True)    # Returning concated dataframe
    return add_to

########################################################################################

#***********                         **********#
#$$$$$$$$$$$ *** Start from here *** $$$$$$$$$$#
#***********                         **********#

clas = 0                           # Enter the class => 1: resistant, 0: suseptible
name = 'SRR2101614'                # enter the SRR accession no of file in string format
isolate_no = name+'_'+str(clas)    # Saving the isolate sample no. using name and class

df = pd.read_csv("C:...\\PROJECT\\Suseptible\\CSV_Susceptible\\"+name+".csv")      # Upload the .csv file; converted from "VCF"
df.head(3)

########################################################################################
# creating three array of zeros in respective length

mut_zero = np.zeros(mutation.shape[1])
gyrmut_zero = np.zeros(gyr_mutation.shape[1])
intermut_zero = np.zeros(intergenic_mutation.shape[1])

print('lenght mut_zero = ', len(mut_zero))
print('lenght gyrmut_zero = ', len(gyrmut_zero))
print('lenght intermut_zero = ', len(intermut_zero))

########################################################################################

#*************                        ***************#
#$$$$$$$$$$$ *** Main code starts here *** $$$$$$$$$$#
#*************                        ***************#

mutation_position = np.array(df)[:,1]   # creating array of postion's from file 'df' 

for j in range(m):                      # range is length of m: (length of 'ref' dataframe)
    
    # for a gene region
    gene_start = ref_num[j,1]           # taking staring position of NCBI Gene range
    gene_stop = ref_num[j,2]            # taking stoping position of NCBI Gene range
    gene_id = ref_num[j,5]              # taking id of NCBI Gene in above mentioned range 
    
    # for an intergenic region
    inter_start = inter_regnum[j,0]     # taking staring position of intergenic region
    inter_stop = inter_regnum[j,1]      # taking stoping position of intergenic region
    inter_id = inter_regnum[j,2]        # taking id of intergenic region in above mentioned range 
    
    '''
    Now, here; updating three array of zeros which was defined earlier
    After updation, they wii consist the mutation 
    for 'every gene-range', 'gyr gene-range' and 'intergenic-range' respectively
    '''
    
    for position in mutation_position:      # for every position in mutation_position array
                # for index,position in enumerate(mutation_position): (code is optional)
        if (position >= gene_start) and (position <= gene_stop): 
            # and(gene_id != 887081) and (gene_id != 887105): (code is optional may be added with above line)
            '''
            this code adds every mutation if it falls under the NCBI gene range
            '''
            mut_zero[j+1] += 1      # if a position caught in a range of particular gene, it will add 1 at that gene
            # Here (j+1) is used because first column is 'isolate_no'
            # mutation_position = mutation_position[index+1:] (code is optional)

            # if (gene_id == 887081) or (gene_id == 887105): 
            '''
            this below code adds every mutation if it falls under the gyrB(887081) and gyrA(887105) gene range
            '''
            # mut_zero[j] += 1  (code is optional)
            if gene_id == 887081:
                for index,num in enumerate(gyr_list[0:2028]):   # range of gyrB in list gyr_list
                    if num == position:
                        gyrmut_zero[index+1] += 1  # 1 is added to 'index'; because the first column is 'isolate_no'
                        #mutation_position = mutation_position[index+1:] (code is optional)
                        break
            elif gene_id == 887105:
                for index,num in enumerate(gyr_list[2028:]):     # range of gyrA in list gyr_list
                    if num == position:
                        gyrmut_zero[index+2029] += 1             # gyrA starts from(index+1+2028)
                        #mutation_position = mutation_position[index+1:]  code is optional)
                        break

        elif (position >= inter_start) and (position <= inter_stop):
            '''
            this code adds every mutation if it falls under intergenic region
            '''
            intermut_zero[j+1] += 1
            # Here (j+1) is used because first column is 'isolate_no'
            # mutation_position = mutation_position[index+1:] (code is optional)

########################################################################################

myvcf_mutation = save_mutation(mut_zero,name, myvcf_mutation, isolate_no, clas, mutate = True)
myvcf_mutation.tail()

########################################################################################

myvcf_gyr = save_mutation(gyrmut_zero, name, myvcf_gyr, isolate_no, clas, gyr = True)
myvcf_gyr.tail()

########################################################################################

myvcf_inter = save_mutation(intermut_zero, name, myvcf_inter, isolate_no, clas, interg = True)
myvcf_inter.tail()

########################################################################################

# Remove the "#" from all below three codes to generate data files
# myvcf_mutation.to_csv((r'C:...\\PROJECT\\mutation_data\\gene_mutation.csv'), index=False)
# myvcf_gyr.to_csv((r'C:...\\PROJECT\\mutation_data\\gyr_mutation.csv'), index=False)
# myvcf_inter.to_csv((r'C:...\\PROJECT\\mutation_data\\interg_mutation.csv'), index=False)

########################################################################################
########################################################################################



