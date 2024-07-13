
import pandas as pd
import numpy as np
'''
This code creates new column: 'Freq' (variant allele frequency),
in original SNPs/InDel file and also creates 
two new files of SNPs/InDel positions where Freq is >= 80%
'''
import copy

#--------------------------------------------------------------------------------------------------#
# Upload Bowtie2 created SNPs file
df_bowtie_snp = pd.read_csv("C:\\...\\SRR1166111_Bowtie2_SNPs.csv")
df_bowtie_snp.head()

df_bowtie_snp_pos = np.array(df_bowtie_snp['POS']) # Creating array of SNPs position
print(len(df_bowtie_snp_pos))

# Upload Bowtie2 created InDel file
df_bowtie_indel = pd.read_csv("C:\\...\\SRR1166111_Bowtie2_INDELs.csv")
df_bowtie_indel.head()

df_bowtie_indel_pos = np.array(df_bowtie_indel['POS'])   # Creating array of InDel position
print(len(df_bowtie_indel_pos))
#--------------------------------------------------------------------------------------------------#

# Creating DataFrame 'Freq' with empty data
data = np.array(['Freq %'])
my_data = pd.DataFrame(columns = data)

# Freq updation for Bowtie2 SNPs file
# Creating array of 9th column 'Sample1' from 'Bowtie2 SNPs file'
Sample1 = np.array(df_bowtie_snp.iloc[:,9])

for i in Sample1:
    '''
    The loop takes only Freq value from each row of 'Sample1' array
    Then it changes those taken Freq value into DataFrame and 
    append into 'my_data'. By this process it extracts all the freq values
    and crates a New column
    '''
    freq = float(list(i.split(':'))[6].split('%')[0])
    
    df_count = pd.DataFrame(data =[[freq]], columns = data)
    my_data = my_data.append(df_count)
    
my_data.reset_index(drop=True, inplace=True)   # Making all indexes in proper order

df2 = df_bowtie_snp.copy()

# creating dataframe in required format by adding new 'Freq %' column
df3 = pd.concat([df2[df2.columns[:7]], my_data, df2[df2.columns[7:]]], axis=1)
df4 = df3[df3['Freq %'] >= 80]  # Selecting only those SNPs positions (Rows) where (Freq is >=80%)
print(df4.shape)
# Downloading the newely crated file
df4.to_csv(
    (r"C:\\...\\SRR1166111_Bowtie2_SNPs_freq_filter.csv"),
           index = False)
df4.head()
#--------------------------------------------------------------------------------------------------#

# Freq updation for Bowtie2 InDel file: Same procedure as the SNPs one file

# Creating DataFrame 'Freq' with empty data
data = np.array(['Freq %'])
my_data = pd.DataFrame(columns = data)

# Creating array of 9th column 'Sample1' from 'Bowtie2 SNPs file'
Sample1 = np.array(df_bowtie_indel.iloc[:,9])

for i in Sample1:
    
    freq = float(list(i.split(':'))[6].split('%')[0])
    
    df_count = pd.DataFrame(data =[[freq]], columns = data)
    my_data = my_data.append(df_count)
    
my_data.reset_index(drop=True, inplace=True)   # Making all indexes in proper order

df5 = df_bowtie_indel.copy()

# creating dataframe in required format by adding new 'Freq %' column
df6 = pd.concat([df5[df5.columns[:7]], my_data, df5[df5.columns[7:]]], axis=1)
# Selecting only those InDel positions (Rows) where (Freq is >=80%)
df7 = df6[df6['Freq %'] >= 80]
print(df7.shape)
# Downloading the newely crated file
df7.to_csv(
    (r"C:\\...\\SRR1166111_Bowtie2_INDELs_freq_filter.csv"),
           index = False)
df7.head()

#----------------------------------------------***----------------------------------------------------#