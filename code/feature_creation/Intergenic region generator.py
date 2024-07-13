
import pandas as pd
import numpy as np


ref = pd.read_excel("...\\PROJECT\\MTb_Genes.xlsx")
ref.head(3)

# creating CSV format of mutation in intergenic region

gen_start = np.array(ref)[:,2]  # taking only start postion of 'ref' data
gen_stop = np.array(ref)[:,3]   # taking only stop postion of 'ref' data

print(gen_start)
print(len(gen_start))
print('')
print(gen_stop)
print(len(gen_stop))

######################################################################################################

intrg_stop = []
intrg_start = []

for ind in range(len(gen_start)):
    
    if (ind < len(gen_start)-1) and (gen_stop[ind] < gen_start[ind+1]-1):
        # creating the end position of intergenic region by taking starting postion of 'ref'
        intrg_stop.append(gen_start[ind+1]-1)   # taking one minus from every start postition of 'ref'
        
        # creating a start position for intergenic region by taking stop postion of 'ref'
        intrg_start.append(gen_stop[ind]+1)     # adding 1 to every stop postition of 'ref'

intrg_start.append(4410930)   # adding the start position (4410930) of intergenic region into array
intrg_stop.append(4411532)    # adding the last potion (4411532) of intergenic region into array

intrg_stop = np.array(intrg_stop)
print(intrg_stop)
print('Length = ',len(intrg_stop))

intrg_start = np.array(intrg_start)  
print(intrg_start)
print('Length = ',len(intrg_start))


intra_name = np.ones(len(intrg_start))   # generating array of ones 
print(len(intra_name))
intra_name=intra_name.astype('U256')     # converting interger type into string type

for i in range(len(intra_name)-1):
    intra_name[i] = 'interg_'+ str(i+1)   # giving name to every intergenic region as 'interg(no)'
print('type = ', type(intra_name[2]))
print(intra_name)
print('Length = ',len(intra_name))


# creating three columns 
column_name1 = ['start']
column_name2 = ['stop']
column_name3 = ['intergenic_region']


# creating three dataframe by taking intergenic region data and naming them by column name
New_mutation_inter1 = pd.DataFrame(data=intrg_start,
                             columns = column_name1)
New_mutation_inter2 = pd.DataFrame(data= intrg_stop,
                             columns = column_name2)
New_mutation_inter3 = pd.DataFrame(data=intra_name,
                             columns = column_name3)

# Concating three newely created dataframe
Intergenic_mut = pd.concat([New_mutation_inter1, New_mutation_inter2, New_mutation_inter3],axis=1)
Intergenic_mut.iloc[-1,-1]= 'interg_2980'  # giving proper name to the outer most postion
print(Intergenic_mut.head())
print(Intergenic_mut.tail())


#downloading dataframe in csv format
Intergenic_mut.to_csv((r'...\\PROJECT\\intergenic_mutation.csv'), index=False)

######################################################################################################