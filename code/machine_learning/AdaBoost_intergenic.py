
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
'''
The code is shown only for one feature file ("intergenic_mutation.csv"), 
which contains mutations in intergenic region
'''
class Data():
    
    def __init__(self):
        pass
    
    def get_data(self,csv_path):
        '''
        Input the data file path as 'csv_path'
        Returns output as Pandas Dataframe
        '''
        return pd.read_csv(csv_path,low_memory=False)
    
    def feature_scaling(self,features):
        '''
        Input a feature data in DataFrame
        Returns Normalized features
        '''
        m,n = features.shape
        for i in range(n):
            if features.iloc[:,i].min() != features.iloc[:,i].max():
                features.iloc[:,i] = (features.iloc[:,i]-features.iloc[:,i].min())/(
                    (features.iloc[:,i].max()-features.iloc[:,i].min()))
        return features
    
    def get_features(self,df,scaling=False, column_name=[]):
        '''
        'df' must be in Pandas Dataframe
        scaling is by-default false for an attribute
        Insert column_name which may be droped:  e.g. column_name=['Isolate_no','Class']
        '''
        features = df
        for col in column_name:
            features = features.drop(col,axis=1) #Drop the columns
        if scaling == True:
            features = Data.feature_scaling(self,features)
        features = np.c_[features]
        return features
    
    def get_targets(self,df,scaling=False, column_name=[]):
        '''
        scaling is by-default false for an attribute
        Insert column_name which will become a target matrix:  e.g. column_name=['Class']
        '''
        targets = df
        for col in column_name:
            targets = targets[col]
        if scaling == True:
            targets = Data.feature_scaling(self,targets)
        return targets
    

def get_plots(p, alogo_gene_name = 'model',initialization = False):
    '''
    Insert 'p' as name of a plot
    Enter 'alogo_gene_name': e.g. 'AdaBoost: All gene data'
    To create a new plot make initialization 'True'
    To update the existing plot make initialization 'False'
    '''
    if initialization == True:
        p.figure(figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')
    else:
        p.plot([0, 1], [0, 1], linestyle=':', lw=2, color='r',label='Random classifier', alpha=.8)
        p.xlim([-0.05, 1.05])
        p.ylim([-0.05, 1.05])
        p.xlabel('False Positive Rate',size=18)
        p.ylabel('True Positive Rate',size=18)
        p.title('Receiver Operator Characteristic Curve for'+ alogo_gene_name, size=22)
        p.legend(loc="lower right")
    return p


def cross_validation(features,targets,classifier,plot_name,n_splits=5):
    '''
    Enter features, targets, classifier, plot_name and n_splits accordingly
    Calculates ROC, AUC, Accuracy and confusion matrix
    Returns AUC, Accuracy, Confusion matrix and Plot for all K-folds
    '''
    aucs,accuracy_list=[],[]
    confusion_mat = {}
    p = get_plots(plot_name,initialization = True)
    i = 1
    np.random.seed(42)
    skf =StratifiedKFold(n_splits,shuffle=True)

    for train_index, test_index in skf.split(features, targets):

        model= classifier.fit(features[train_index], targets[train_index])
        prediction = model.predict(features[test_index])
        score = accuracy_score(prediction,targets[test_index])
        prob = model.predict_proba(features[test_index])
        fpr, tpr, thresholds = roc_curve(targets[test_index], prob[:,1])  # Calculating ROC
        roc_auc = auc(fpr, tpr)  # Calculating AUC
        aucs.append(roc_auc)
        p.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        # Claculating Confusion matrix
        confusion_mat['acc'+str(i)+'_'+str(score)] = confusion_matrix(targets[test_index],
                                                                  prediction,labels=[0,1])
        accuracy_list.append(score)
        i += 1
    return aucs, accuracy_list, confusion_mat, p


if __name__ == '__main__':
    
    data = Data()
    csv_path = "C:\\...\\mutation_data\\intergenic_mutation.csv"
    all_gene_df = data.get_data(csv_path)
    X = data.get_features(all_gene_df,scaling=False, column_name=['Isolate_no','Class'])
    print("Shape of feature matrix : ",X.shape)
    y = data.get_targets(all_gene_df, column_name=['Class'])
    print("Shape of target matrix : ",y.shape)

    #*****************************   AdaBoost DT: Intergenic data   *****************************#
    
    classifier = AdaBoostClassifier(n_estimators=20,learning_rate=0.25)
    aucs, accuracy_list, confusion_mat, plot = cross_validation(X,y,classifier,plt)
    print("\nMean Accuracy of Adaboost DT intergenic data: ", np.array(accuracy_list).mean())
    print("Mean Area under the curve (AUC) for Adaboost DT intergenic : ",np.array(aucs).mean())
    print("Mean Standard deviation of AUC for Adaboost DT intergenic : ",np.std(aucs))
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n")
    
    AUC = pd.DataFrame(data=aucs, columns = ['AdaBoost_DT_intergenic'])
    AUC.to_csv((r'C:\\...\\AdaBoost_DT_intergenic.csv'), index=False)
    
    plot = get_plots(plot, alogo_gene_name = ' Adaboost DT: Intergenic data',initialization = False)
    plot.savefig(r'C:\\...\\AdaBoost_DT_intergenic.png',dpi=200)
    plot.show()

    # *****************************   AdaBoost LR: Intergenic data   ***************************** #
    
    lgr = LogisticRegression(solver='lbfgs',random_state=42)
    classifier = AdaBoostClassifier(n_estimators=80,base_estimator=lgr,
                                    learning_rate=0.25)
    aucs, accuracy_list, confusion_mat, plot = cross_validation(X,y,classifier,plt)
    print("\nMean Accuracy of Adaboost LR intergenic data: ", np.array(accuracy_list).mean())
    print("Mean Area under the curve (AUC) for Adaboost LR intergenic : ",np.array(aucs).mean())
    print("Mean Standard deviation of AUC for Adaboost LR intergenic : ",np.std(aucs))
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n")
    
    AUC = pd.DataFrame(data=aucs, columns = ['AdaBoost_LR_intergenic'])
    AUC.to_csv((r'C:\\...\\AdaBoost_LR_intergenic.csv'), index=False)
    
    plot = get_plots(plot, alogo_gene_name = ' Adaboost LR: Intergenic data',initialization = False)
    plot.savefig(r'C:\\...\\AdaBoost_LR_intergenic.png',dpi=200)
    plot.show()

    # *****************************  AdaBoost SVM Linear kernel: Intergenic data  *****************************

    svc = SVC(probability=True, kernel='linear',random_state = 42) # Selecting kernel as 'Linear'
    classifier = AdaBoostClassifier(n_estimators=20, base_estimator=svc,
                                    learning_rate=0.075)
    aucs, accuracy_list, confusion_mat, plot = cross_validation(X,y,classifier,plt)
    print("\nMean Accuracy of Adaboost SVM-Linear intergenic data: ", np.array(accuracy_list).mean())
    print("Mean Area under the curve (AUC) for Adaboost SVM-Linear intergenic : ",np.array(aucs).mean())
    print("Mean Standard deviation of AUC for Adaboost SVM-Linear intergenic : ",np.std(aucs))
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n")
    
    AUC = pd.DataFrame(data=aucs, columns = ['AdaBoost_SVM-Linear_intergenic'])
    AUC.to_csv((r'C:\\...\\AdaBoost_SVM-Linear_intergenic.csv'), index=False)
    
    plot = get_plots(plot, alogo_gene_name = ' Adaboost SVM-Linear: Intergenic data',initialization = False)
    plot.savefig(r'C:\\...\\AdaBoost_SVM-Linear_intergenic.png',dpi=200)
    plot.show()

    # ***************************** AdaBoost SVM rbf kernel: Intergenic data  ***************************** #
    
    svc = SVC(probability=True, gamma='auto', kernel='rbf',random_state = 42) # Selecting kernel as 'rbf'
    classifier = AdaBoostClassifier(n_estimators=80, base_estimator=svc,
                                    learning_rate=0.25)
    aucs, accuracy_list, confusion_mat, plot = cross_validation(X,y,classifier,plt)
    print("\nMean Accuracy of Adaboost SVM-rbf intergenic data: ", np.array(accuracy_list).mean())
    print("Mean Area under the curve (AUC) for Adaboost SVM-rbf intergenic : ",np.array(aucs).mean())
    print("Mean Standard deviation of AUC for Adaboost SVM-rbf intergenic : ",np.std(aucs))
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n")
    
    AUC = pd.DataFrame(data=aucs, columns = ['AdaBoost_SVM-rbf_intergenic'])
    AUC.to_csv((r'C:\\...\\AdaBoost_SVM-rbf_intergenic.csv'), index=False)
    
    plot = get_plots(plot, alogo_gene_name = ' Adaboost SVM-rbf: Intergenic data',initialization = False)
    plot.savefig(r'C:\\...\\AdaBoost_SVM-rbf_intergenic.png',dpi=200)
    plot.show()

#-----------------------------------------------     ***     -----------------------------------------------------#