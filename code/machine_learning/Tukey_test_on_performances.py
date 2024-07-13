
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#...................................................................................................#

# creating dataframe using results (AUC) of top 8 performing model+features for Moxifloxacin drug MTb data

df_moxi = pd.DataFrame({'score': [0.930555556, 0.975694444, 0.96875, 0.949652778, 0.986111111,
                             0.921875, 0.982638889, 0.972222222, 0.930555556, 0.982638889,
                              0.928819444, 0.963541667, 0.972222222, 0.944444444, 0.972222222,
                              0.890625, 0.984375, 0.984375, 0.925347222, 0.987847222,
                              0.921875, 0.972222222, 0.963541667, 0.940972222, 0.973958333,
                              0.890625, 0.979166667, 0.972222222, 0.939236111, 0.989583333,
                              0.901041667, 0.975694444, 0.973958333, 0.939236111, 0.979166667,
                              0.925347222, 0.954861111, 0.973958333, 0.9375, 0.973958333],
                   'group': np.repeat(['AdB+M3+F1', 'AdB+M2+F4', 'Adb+M4+F1', 
                                       'AdB+M1+F1', 'AdB+M3+F4', 'AdB+M1+F4',
                                      'AdB+M2+F1', 'AdB+M4+F4'], repeats=5)}) 

# perform Tukey's test
tukey_moxi = pairwise_tukeyhsd(endog=df_moxi['score'],
                          groups=df_moxi['group'],
                          alpha=0.05)
#display results
print(tukey_moxi)
#...................................................................................................#

# creating dataframe using results (AUC) of all models for Ofloxacin drug MTb data

df_ofl = pd.DataFrame({'score': [0.796305931, 0.848074922, 0.741198603, 0.752754636, 0.768341844,
                                 0.798647242, 0.849635796, 0.760279495, 0.760816985, 0.75517334,
                                 0.828563996, 0.875650364, 0.768610589, 0.792260145, 0.844127923,
                                 0.91896462, 0.965530697, 0.859446385, 0.96640688, 0.956732061],
                   'group': np.repeat(['AdaBoost_SVM-rbf_all_gene', 'AdaBoost_SVM-Linear_all_gene', 
                                       'AdaBoost_LR_all_gene', 'AdaBoost_DT_all_gene'], repeats=5)})

# perform Tukey's test
tukey_ofl = pairwise_tukeyhsd(endog=df_ofl['score'],
                          groups=df_ofl['group'],
                          alpha=0.05)

#display results
print(tukey_ofl)
#...................................................................................................#

# creating dataframe using results (AUC) of top 8 performing model+features for Ciprofloxacin drug MTb data

df_cip = pd.DataFrame({'score': [1, 0.982248521, 0.982248521, 0.982248521, 0.965277778,
                                 1, 0.979289941, 0.979289941, 0.99408284, 0.954861111,
                                 0.98816568, 0.979289941, 0.970414201, 0.99408284, 0.972222222,
                                 1, 0.973372781, 0.979289941, 0.99408284, 0.9375,
                                 0.98816568, 0.99704142, 0.961538462, 1, 0.895833333,
                                 0.99408284, 0.923076923, 0.98816568, 0.917159763, 0.972222222,
                                 0.893491124, 0.917159763, 0.946745562, 0.964497041, 0.951388889,
                                 0.899408284, 0.893491124, 0.946745562, 0.970414201, 0.944444444],
                   'group': np.repeat(['AdB+M1+F4', 'AdB+M1+F3', 'AdB+M4+F3', 
                                       'AdB+M3+F3', 'AdB+M2+F3', 'AdB+M1+F1',
                                      'AdB+M2+F1', 'AdB+M2+F4'], repeats=5)})

# perform Tukey's test
tukey_cip = pairwise_tukeyhsd(endog=df_cip['score'],
                          groups=df_cip['group'],
                          alpha=0.05)

#display results
print(tukey_cip)
#---------------------------------------------    ***    -------------------------------------------------#