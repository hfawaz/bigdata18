import numpy as np 
import operator
import utils 

def knn(x_train,y_train,x_test, k, distance_algorithm = 'dtw'):
    """
    Performs knn and returns the predicted values. 
    :param k: The number of neighbors to consider.
    """
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the corresponding parameters 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    y_pred = []
    # loop through the test 
    for i in range(x_test.shape[0]):
        # get the neighbors 
        neighbors = get_neighbors(x_train,x_test[i],k,dist_fun,dist_fun_params)
        # predict the label with majority vote over the neighbors 
        predicted_label = get_predicted_label(neighbors, y_train)
        y_pred.append(predicted_label)
    # transform to numpy array 
    y_pred = np.array(y_pred)
    return y_pred

def get_neighbors(x_train, x_test_instance, k, dist_fun, dist_fun_params, 
                  pre_computed_matrix=None, index_test_instance=None,
                  return_distances = False): 
    """
    Given a test instance, this function returns its neighbors present in x_train
    NB: If k==0 zero it only returns the distances
    """
    distances = []
    # loop through the training set 
    for i in range(len(x_train)): 
        # calculate the distance between the test instance and each training instance
        if pre_computed_matrix is None: 
            dist , _ = dist_fun(x_test_instance, x_train[i],**dist_fun_params)
        else: 
            # do not re-compute the distance just get it from the precomputed one
            dist = pre_computed_matrix[i,index_test_instance]
        # add the index of the current training instance and its corresponding distance 
        distances.append((i, dist))
    # if k (nb_neighbors) is zero return all the items with their distances 
    # NOT SORTED 
    if k==0: 
        if return_distances == True: 
            return distances
        else:
            print('Not implemented yet')
            exit()
    # sort list by specifying the second item to be sorted on 
    distances.sort(key=operator.itemgetter(1))
    # else do return only the k nearest neighbors
    neighbors = []
    for i in range(k): 
        if return_distances == True: 
            # add the index and the distance of the k nearest instances from the train set 
            neighbors.append(distances[i])
        else:
            # add only the index of the k nearest instances from the train set 
            neighbors.append(distances[i][0])
        
    return neighbors

def get_predicted_label(neighbors, y_train):
    """
    Returns the label by a majority vote from the neighbors
    """
    class_votes = {}
    # loop through all neighbors and their labels
    for idx_neighbor in neighbors: 
        # get the label of this neighbor 
        curr_label = y_train[idx_neighbor]
        # increment the label count 
        if curr_label in class_votes:
            class_votes[curr_label] += 1
        else: 
            class_votes[curr_label] = 1
    # now sort the dictionary 
    sorted_votes = sorted(class_votes.items(), 
                          key=operator.itemgetter(1), reverse=True)
    # return the first label 
    return sorted_votes[0][0]
   
    