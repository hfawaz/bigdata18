 # this is a transfer learning example 
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES as ALL_DATASET_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS

import time 
import keras 
import numpy as np
import sys

from utils.utils import create_directory
from utils.utils import read_all_datasets
from utils.utils import save_logs
from utils.utils import transform_labels
from utils.utils import visualize_transfer_learning
from utils.utils import get_random_initial_for_kmeans
import utils
import operator
from knn import get_neighbors

from datareduce import reduce

import pandas as pd

def read_data_from_dataset(use_init_clusters=True):

	x_train = datasets_dict[dataset_name][0]
	y_train = datasets_dict[dataset_name][1]
	x_test = datasets_dict[dataset_name][2]
	y_test = datasets_dict[dataset_name][3]

	nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
	# make the min to zero of labels
	y_train, y_test = transform_labels(y_train, y_test)

	classes, classes_counts = np.unique(y_train, return_counts=True)

	if len(x_train.shape) == 2:  # if univariate
		# add a dimension to make it multivariate with one dimension
		x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
		x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

	# maximum number of prototypes which is the minimum count of a class
	max_prototypes = min(classes_counts.max() + 1,
						 MAX_PROTOTYPES_PER_CLASS + 1)
	init_clusters = None

	if use_init_clusters == True:
		# set the array that contains the initial clusters for k-means
		init_clusters = get_random_initial_for_kmeans(x_train, y_train,
													  max_prototypes, nb_classes)
	return x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, init_clusters

def build_model(input_shape, nb_classes, pre_model=None):
	input_layer = keras.layers.Input(input_shape)

	conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
	conv1 = keras.layers.normalization.BatchNormalization()(conv1)
	conv1 = keras.layers.Activation(activation='relu')(conv1)

	conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
	conv2 = keras.layers.normalization.BatchNormalization()(conv2)
	conv2 = keras.layers.Activation('relu')(conv2)

	conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
	conv3 = keras.layers.normalization.BatchNormalization()(conv3)
	conv3 = keras.layers.Activation('relu')(conv3)

	gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

	output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer)

	if pre_model is not None:

		for i in range(len(model.layers)-1):
			model.layers[i].set_weights(pre_model.layers[i].get_weights())

	model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(),
		metrics=['accuracy'])

	return model

def train(pre_model=None):
	# read train, val and test sets 
	x_train = datasets_dict[dataset_name_tranfer][0]
	y_train = datasets_dict[dataset_name_tranfer][1]

	y_true_val = None 
	y_pred_val = None

	x_test = datasets_dict[dataset_name_tranfer][-2]
	y_test = datasets_dict[dataset_name_tranfer][-1]
	

	mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

	nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

	# make the min to zero of labels
	y_train,y_test = transform_labels(y_train,y_test)

	# save orignal y because later we will use binary
	y_true = y_test.astype(np.int64)

	# transform the labels from integers to one hot vectors
	y_train = keras.utils.to_categorical(y_train, nb_classes)
	y_test = keras.utils.to_categorical(y_test, nb_classes)

	if len(x_train.shape) == 2: # if univariate 
		# add a dimension to make it multivariate with one dimension 
		x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
		x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

	start_time = time.time()
	# remove last layer to replace with a new one 
	input_shape = (None,x_train.shape[2])
	model = build_model(input_shape, nb_classes,pre_model)

	if verbose == True: 
		model.summary()

	# b = model.layers[1].get_weights()

	hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
		verbose=verbose, validation_data=(x_test,y_test), callbacks=callbacks)

	# a = model.layers[1].get_weights()

	# compare_weights(a,b)

	model = keras.models.load_model(file_path)

	y_pred = model.predict(x_test)
	# convert the predicted from binary to integer 
	y_pred = np.argmax(y_pred , axis=1)

	duration = time.time()-start_time

	df_metrics = save_logs(write_output_dir, hist, y_pred, y_true,
						   duration,lr=True, y_true_val=y_true_val,
						   y_pred_val=y_pred_val)

	print('df_metrics')
	print(df_metrics)

	keras.backend.clear_session()

def reduce_function(reduce_algorithm_name, x_train, y_train, C, init_clusters_per_class):
	if reduce_algorithm_name == 'dtw_dba_kmeans':
		return reduce(x_train, y_train, C, clustering_algorithm='kmeans',
					  averaging_algorithm='dba', distance_algorithm='dtw',
					  init_clusters_per_class=init_clusters_per_class), 'dtw'

root_dir = '/b/home/uha/hfawaz-datas/dl-tsc/'

results_dir = root_dir+'results/fcn/'

batch_size = 16
nb_epochs = 2000
verbose = False

write_dir_root = root_dir+'/transfer-learning-results/'

if sys.argv[1] == 'transfer_learning':
	# loop through all archives
	for archive_name in ARCHIVE_NAMES:
		# read all datasets
		datasets_dict = read_all_datasets(root_dir,archive_name)
		# loop through all datasets
		for dataset_name in ALL_DATASET_NAMES:
			# get the directory of the model for this current dataset_name
			output_dir = results_dir+archive_name +'/'+dataset_name+'/'
			# loop through all the datasets to transfer to the learning
			for dataset_name_tranfer in ALL_DATASET_NAMES:
				# check if its the same dataset
				if dataset_name == dataset_name_tranfer:
					continue
				# set the output directory to write new transfer learning results
				write_output_dir = write_dir_root+archive_name+'/'+dataset_name+\
					'/'+dataset_name_tranfer+'/'
				write_output_dir = create_directory(write_output_dir)
				if write_output_dir is None:
					continue
				print('Tranfering from '+dataset_name+' to '+dataset_name_tranfer)
				# load the model to transfer to other datasets
				pre_model = keras.models.load_model(output_dir+'best_model.hdf5')
				# output file path for the new tranfered re-trained model
				file_path = write_output_dir+'best_model.hdf5'
				# callbacks
				# reduce learning rate
				reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
					min_lr=0.0001)
				# model checkpoint
				model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
					save_best_only=True)
				callbacks=[reduce_lr,model_checkpoint]

				train(pre_model)

elif sys.argv[1] == 'train_fcn_scratch':
	# loop through all archives
	for archive_name in ARCHIVE_NAMES:
		# read all datasets
		datasets_dict = read_all_datasets(root_dir, archive_name)
		for dataset_name_tranfer in ALL_DATASET_NAMES:
			# get the directory of the model for this current dataset_name
			write_output_dir = results_dir + archive_name+'_for_git' + '/' + dataset_name_tranfer + '/'
			# set model output path
			file_path = write_output_dir + 'best_model.hdf5'
			# create directory
			create_directory(write_output_dir)
			# reduce learning rate
			reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
														  patience=50,min_lr=0.0001)
			# model checkpoint
			model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
															   save_best_only=True)
			callbacks=[reduce_lr,model_checkpoint]

			train()

elif sys.argv[1]=='visualize_transfer_learning':
	visualize_transfer_learning(root_dir)

elif sys.argv[1] == 'compare_datasets':
	nb_prototype = 1
	tot_classes = 0
	tot_x_train = []
	tot_y_train = []
	nb_neighbors = 84
	distance_algorithm = 'dtw'
	# get the distance function
	dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
	# get the corresponding parameters
	dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
	# loop the archive names
	for archive_name in ARCHIVE_NAMES:
		# read all the datasets
		datasets_dict = read_all_datasets(root_dir, archive_name)
		# loop through all the dataset names
		for dataset_name in ALL_DATASET_NAMES:
			print('dataset_name: ', dataset_name)
			# read the train and test
			x_train, y_train, x_test, y_test, nb_classes, classes, max_prototypes, \
			init_clusters = read_data_from_dataset(use_init_clusters=True)
			tot_classes += nb_classes
			# get init of the clusters for one prototype and the 0th iteration
			init_clusters_per_class = init_clusters[nb_prototype][0]
			# reduce the dataset
			train_set, _ = reduce_function('dtw_dba_kmeans', x_train, y_train, nb_prototype,
										   init_clusters_per_class)
			# read the training instances and their labels
			x_train, y_train = train_set
			# create the new y_train composed of the dataset_name
			y_train = [dataset_name for _ in range(len(y_train))]
			# put the new train in tot train set
			tot_x_train = tot_x_train + x_train
			tot_y_train = tot_y_train + y_train

	classes = np.unique(tot_y_train)
	nb_classes = len(classes)

	columns = [('K_' + str(i)) for i in range(1, nb_neighbors + 1)]
	neighbors = pd.DataFrame(data=np.zeros((nb_classes, nb_neighbors),
										   dtype=np.str_), columns=columns, index=classes)

	# to numpyyyyyy
	tot_y_train = np.array(tot_y_train)
	tot_x_train = np.array(tot_x_train)

	# this is also a lopp over the names of the datasets
	for c in classes:
		# get the x_train without the test instances
		x_train = tot_x_train[np.where(tot_y_train != c)]
		# get the y_train without the test instances
		y_train = tot_y_train[np.where(tot_y_train != c)]
		# get the x_test instances
		x_test = tot_x_train[np.where(tot_y_train == c)]
		# init the distances
		distances = []
		# loop through each test instances
		for x_test_instance in x_test:
			# get the nearest neighbors
			distance_neighbors = get_neighbors(x_train, x_test_instance,
											   0, dist_fun, dist_fun_params, return_distances=True)
			# concat the distances
			distances = distances + distance_neighbors
		# sort list by specifying the second item to be sorted on
		distances.sort(key=operator.itemgetter(1))
		# to numpy array the second item only (the label)
		distances = np.array([y_train[distances[i][0]] \
							  for i in range(len(distances))])
		# aggregate the closest datasets
		# this is useful if two datasets are in the k nearest neighbors
		# more than once because they have more than one similar class
		distances = pd.unique(distances)
		# leave only the k nearest ones
		for i in range(1, nb_neighbors + 1):
			# get label of the neighbor
			label = distances[i - 1]
			# put the label
			neighbors.loc[c]['K_' + str(i)] = label

	neighbors.to_csv(root_dir + 'similar-datasets.csv')
	print(neighbors.to_string())