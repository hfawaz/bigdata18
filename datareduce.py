import numpy as np 
import utils 

def reduce(x_train, y_train, C, init_clusters_per_class, 
           clustering_algorithm = 'kmeans', averaging_algorithm = 'dba',
           distance_algorithm='dtw'):
    """
    Reduces the x_train using an averaging method applied on the result of clustering
    :param x_train: The time series to be reduced.
    :param y_train: The corresponding labels of the time series to be reduced. 
    :param C: The number of exemplars per class.
    :param init_clusters_per_class: the initial clusters for k-means per class
    :param nb_iterations: Nbr of iterations of the clustering algorithm to converge. 
    :param max_iter: Nbr of iterations of the averaging algorithm.
    """
    # get the averaging method (check constants.py)
    avg_method = utils.constants.AVERAGING_ALGORITHMS[averaging_algorithm]
    # get the clustering algorithm (check constants.py)
    cluster_method = utils.constants.CLUSTERING_ALGORITHMS[clustering_algorithm]
    # get all unique classes 
    classes = np.unique(y_train)
    # get the number of classes 
    nb_classes = len(classes)
    # initialize list of condensed time series per class 
    # for each class we have a list of the reduced time series 
    condensed_data = [[] for i in range(nb_classes)]
    # loop through classes
    for c in range(nb_classes): 
        # get time series of current class 
        c_x_train = x_train[np.where(y_train==c)]
        # limit the nb_prototypes 
        nb_prototypes_per_class = min(C, len(c_x_train)) 
        # cluster the time series
        _,affect = cluster_method(c_x_train, nb_prototypes_per_class, 
                                  init_clusters = init_clusters_per_class[c],
                                  averaging_algorithm = averaging_algorithm, 
                                  distance_algorithm = distance_algorithm)
        # loop through all clusters 
        for i in range(nb_prototypes_per_class):
            # get the group of time series affected to cluster index i 
            data_to_condense = [c_x_train[j] for j, x in enumerate(affect) if x == i]
            # check if this cluster center has no affected 
            if len(data_to_condense) == 0: 
                # if so do not average 
                print('Empty cluster ',i, ' for class ',c,' with ',len(c_x_train),' examples')
                continue
            # condense the data 
            c_x_train_reduced = avg_method(data_to_condense, 
                                           distance_algorithm=distance_algorithm)
            # add the average to the list of condensed or reduced data for class c
            condensed_data[c].append(c_x_train_reduced)
         
    return reconstruct_reduced_trainset(condensed_data)

# it is the same as reduce but for one class 
def reduce_per_class(curr_dir, c_x_train, c_y_train, C, c_init_clusters, class_c,
           clustering_algorithm = 'kmeans', averaging_algorithm = 'dba',
           distance_algorithm='dtw'):
    """
    Reduces the x_train using an averaging method applied on the result of clustering
    :param x_train: The time series to be reduced.
    :param y_train: The corresponding labels of the time series to be reduced. 
    :param C: The number of exemplars per class.
    :param c_init_clusters: the initial clusters for k-means for the class_c
    """
    # get the averaging method (check constants.py)
    avg_method = utils.constants.AVERAGING_ALGORITHMS[averaging_algorithm]
    # get the clustering algorithm (check constants.py)
    cluster_method = utils.constants.CLUSTERING_ALGORITHMS[clustering_algorithm]
    # list that contains the prototypes for class_c
    condensed_data_c = [] 
    # limit the nb_prototypes 
    nb_prototypes_per_class = min(C, len(c_x_train)) 
    # cluster the time series
    _,affect = cluster_method(curr_dir, c_x_train, nb_prototypes_per_class,
                              init_clusters = c_init_clusters,
                              averaging_algorithm = averaging_algorithm, 
                              distance_algorithm = distance_algorithm)
    # loop through all clusters 
    for i in range(nb_prototypes_per_class):
        # get the group of time series affected to cluster index i 
        data_to_condense = [c_x_train[j] for j, x in enumerate(affect) if x == i]
        # check if this cluster center has no affected 
        if len(data_to_condense) == 0: 
            # if so do not average 
            print('Empty cluster ',i, ' for class ',class_c,' with ',
                  len(c_x_train),' examples')
            continue
        # condense the data 
        c_x_train_reduced = avg_method(data_to_condense, 
                                       distance_algorithm=distance_algorithm)
        # add the average to the list of condensed or reduced data for class c
        condensed_data_c.append(c_x_train_reduced)
         
    return condensed_data_c

def reconstruct_reduced_trainset(condensed_data):
    nb_classes = len(condensed_data)
    # construc the reduced_x_train and the reduced_y_train 
    reduced_x_train = []
    reduced_y_train = []
    # loop through all the classes
    for c in range(nb_classes): 
        # add the condensed data of the current class to the new x_train
        reduced_x_train.extend(condensed_data[c])
        C = len(condensed_data[c])
        # add the corresponding classes of the condensed data
        reduced_y_train.extend([c for i in range(C)])
        
    # return the list of condensed data of all classes        
    return reduced_x_train, reduced_y_train
    
def reduce_data_random(x_train, y_train, C): 
    """
    Reduces the data by randomly selecting C instances from each class
    """
    # get all unique classes 
    classes = np.unique(y_train)
    # get the number of classes 
    nb_classes = len(classes)
    # initialize list of condensed time series per class 
    # for each class we have a list of the reduced time series 
    condensed_data = [[] for i in range(nb_classes)]
    # loop through classes
    for c in range(nb_classes): 
        # get time series of current class 
        c_x_train = x_train[np.where(y_train==c)]
        # limit the nb_prototypes 
        nb_prototypes_per_class = min(C, len(c_x_train)) 
        # choose C random indices 
        random_idx = np.random.permutation(len(c_x_train))[:nb_prototypes_per_class]
        # get the condensed data based on the random indices 
        condensed_data[c] = [c_x_train[i] for i in random_idx]
    
    return reconstruct_reduced_trainset(condensed_data)
    
        
    
        
    