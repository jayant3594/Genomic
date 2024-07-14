# Genomic

Treatment of infections is compromised worldwide by the emergence of bacteria that are resistant to multiple antibiotics. Over the years, continued selective pressure by different drugs has resulted in organisms bearing additional kinds of resistance mechanisms that led
to multidrug resistance (MDR).

The most notable example is Mycobacterium tuberculosis (MTb) which is responsible for disease Tuberculosis (TB). Major challenges come in the path of controlling the TB is timely detection of drug-resistant bacteria. The existing methods are time-consuming as MTb doubling time is 24 hours, and therefore it takes 6 weeks before one can detect its resistivity to antibiotics. 

The drug-resistivity in bacteria is caused by genetic level mutation also known as Single nucleotide polymorphism (SNPs) and InDels. 

Newer sequence-based techniques such as Whole genome sequencing (WGS) are very effective and capable in the identification of SNPs and InDels. They are faster compared to conventional techniques. 

Recently, the application of machine learning algorithms to these WGS data has made it possible to detect the drug-resistant bacteria in just a few hours. 

In this study, the WGS data of MTb against three second line Fluoroquinolone drugs Moxifloxacin, Ofloxacin and Ciprofloxacin is collected. The SNPs and InDels are identified in them by referencing with MTb H37Rv genome. 

Using SNPs data; four different feature sets are created mainly based on all gene mutation, Intergenic region mutation, Gyrase B/A gene mutation and whole genome mutation. 

The exploratory data analysis is performed to extract the crucial information contained in the data. Different machine learning
classification algorithms are applied to all the feature sets along with various dimensionality reduction techniques. 

The performance analysis is done to identify the best set of feature and algorithm using techniques such as Tukey’s test.