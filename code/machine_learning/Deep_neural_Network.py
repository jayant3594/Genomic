
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)
'''
Source: The initial coding part is courtesy to the course 
'Neural Networks and Deep Learning' by DeepLearning.AI
Instructor: Andrew Ng
'''
#--------------------------------   Neural network part   -----------------------------------#
# FUNCTION: initialize_parameters_deep

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(57)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        np.random.seed(l)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

# FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def sigmoid(Z):
    """
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    return dZ

def sigmoid_backward(dA, cache):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    return dZ

# FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implementing [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,  parameters['W' + str(l)],  parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    # Implementing LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,  parameters['W' + str(L)],  parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
       
    return AL, caches

# FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: 
    containing 0 if Susceptible isolate, 1 if Resistant), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    # Computing loss from aL and y.
    cost = (-1/m)* np.sum(np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
   
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost

# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implementing the linear portion of backward propagation for a single layer (layer l)
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)* np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
   
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implementing the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


# FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implementing the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if Susceptible isolate, 1 if Resistant)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

# FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate):
    """
    Updating parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

# FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if Susceptible isolate, 1 if Resistant), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
   
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
#     # plot the cost
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per hundreds)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
    
    return parameters

#--------------------------------   Data input part   -----------------------------------#

class Data():
    '''
    Created a Class object for initial Data processing
    '''
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
        Insert column_name which may be droped:  e.g. column_name=['isolate_no','class']
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
        Insert column_name which will become a target matrix:  e.g. column_name=['class']
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

def cross_validation(features,targets,layers_dims, plot_name, n_splits=5):
    '''
    Enter features, targets,layers_dims , plot_name and n_splits accordingly
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
        
        parameters = L_layer_model(
            features[train_index].T, 
            targets[train_index].T, 
            layers_dims, 
            num_iterations = 2500, print_cost = True)
        
        prediction, cashes = L_model_forward(features[test_index].T, parameters)
        
        fpr, tpr, thresholds = roc_curve(
            np.reshape(targets[test_index], len(test_index)), 
            np.reshape(prediction,len(test_index))
        )
        # Calculating ROC
        roc_auc = auc(fpr, tpr)  # Calculating AUC
        aucs.append(roc_auc)
        p.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        score = accuracy_score(prediction.T ,targets[test_index])
    
        # Claculating Confusion matrix
        confusion_mat['acc'+str(i)+'_'+str(score)] = confusion_matrix(targets[test_index],
                                                                  prediction.T,labels=[0,1])
        accuracy_list.append(score)
        i += 1
    return aucs, accuracy_list, confusion_mat, p

#--------------------------------   Main program   -----------------------------------#

if __name__ == '__main__':
    
    data = Data()
    # Uploading data
    csv_path = "C:\\...\\all_gene_mutation.csv"
    all_gene_df = data.get_data(csv_path)
    X = data.get_features(all_gene_df,scaling=True, column_name=['Isolate_no','Class'])
    print("Shape of feature matrix : ",X.shape)
    y = data.get_targets(all_gene_df, column_name=['Class'])
    y = np.array([y]).T
    print("Shape of target matrix : ",y.shape)
    
    layers_dims = [3906, 20, 7, 5, 1] #  4-layer model
    aucs, accuracy_list, confusion_mat, plot = cross_validation(X,y,layers_dims, plt, n_splits=5)

    print("\nMean Accuracy of Deep Neural Network All gene data: ", np.array(accuracy_list).mean())
    print("Mean Area under the curve (AUC) for Deep Neural Network All gene data : ",np.array(aucs).mean())
    print("Mean Standard deviation of AUC for Deep Neural Network All gene data : ",np.std(aucs))
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n")

    AUC = pd.DataFrame(data=aucs, columns = ['Deep_Neural_Network_all_gene'])
    #     AUC.to_csv((r'C:\\...\\AdaBoost_DT_all_gene.csv'), index=False)

    plot = get_plots(plot, alogo_gene_name = ' Deep Neural Network: All gene data',initialization = False)
    plot.savefig(r'C:\\...\\DNN_all_gene.png',dpi=200)
    plot.show()

    print(accuracy_list)
    print('\nConfusion Matrix :')
    print(*confusion_mat.items(),sep = "\n\n")
    print(aucs)
#-----------------------------------------   ***   -------------------------------------------#