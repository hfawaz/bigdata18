from builtins import print

import numpy as np
import pandas as pd 
import matplotlib
from pandas.tests.extension import decimal

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx 

import seaborn as sns 

import utils

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS
from utils.constants import NB_ITERATIONS

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


from scipy.io import loadmat

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

def create_path(root_dir,classifier_name, archive_name):
    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+'/'
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

def read_dataset(root_dir,archive_name,dataset_name):
    datasets_dict = {}
    file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'+dataset_name
    x_train, y_train = readucr(file_name+'_TRAIN')
    x_test, y_test = readucr(file_name+'_TEST')
    datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
        y_test.copy())

    return datasets_dict

def get_random_initial_for_kmeans(x_train,y_train, max_prototypes, nb_classes):

    res = np.zeros((max_prototypes,NB_ITERATIONS,nb_classes,max_prototypes,
                    x_train.shape[1],x_train.shape[2]),dtype=np.float64)
    for i in range(1,max_prototypes):
        for j in range(NB_ITERATIONS):
            for c in range(nb_classes):
                c_x_train = x_train[np.where(y_train==c)]
                num_clusters = min(i,len(c_x_train))
                clusterIdx = np.random.permutation(len(c_x_train))[:num_clusters]
                clusters = [c_x_train[cl] for cl in clusterIdx]
                for cc in range(num_clusters):
                    res[i,j,c,cc,:,:]=np.array(clusters[cc])
    return res

def read_all_datasets(root_dir,archive_name):
    datasets_dict = {}

    dataset_names_to_sort = []

    for dataset_name in DATASET_NAMES:
        root_dir_dataset =root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        file_name = root_dir_dataset+dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN')
        x_test, y_test = readucr(file_name+'_TEST')

        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())

        dataset_names_to_sort.append((dataset_name,len(x_train)))

    dataset_names_to_sort.sort(key=operator.itemgetter(1))

    for i in range(len(DATASET_NAMES)):
        DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

def calculate_metrics(y_true, y_pred,duration,y_true_val=None,y_pred_val=None): 
    res = pd.DataFrame(data = np.zeros((1,4),dtype=np.float), index=[0], 
        columns=['precision','accuracy','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)
    
    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    return res

def transform_labels(y_train,y_test):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train,y_test),axis =0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test

def add_results_from_bake_off(df_res,df_res_bake_off,
    classifiers_to_add=['COTE','ST','BOSS','EE','PF','DTW_R1_1NN']):
    df_res_bake_off_to_add = df_res_bake_off.loc[\
        df_res_bake_off['classifier_name'].isin(classifiers_to_add) \
            & df_res_bake_off['dataset_name'].isin(DATASET_NAMES)]
    pd_bake_off = pd.concat([df_res,df_res_bake_off_to_add],sort=False)
    return pd_bake_off

def generate_results_csv_from_bake_off(root_dir='/mnt/nfs/casimir/', bake_off_only=False):
    if bake_off_only==False:
        df_temp = pd.read_csv(root_dir+'results-bake-off-train-test-split-ucr.csv',
            index_col=0)
    else:
        # change name of file to results-bake-off.csv if you bake-off cd diagrams on 100 splits
        df_temp = pd.read_csv(root_dir+'results-bake-off-train-test-split-ucr.csv',index_col=0)
    classifiers  = np.array(df_temp.columns)
    m=len(classifiers)
    classifiers = classifiers.reshape(1,m)
    datasets = np.array(df_temp.index)
    d = len(datasets)
    datasets = datasets.reshape(1,d)
    acc_data = np.array(df_temp).reshape(d*m,1)
    cc = np.repeat(classifiers,d,axis=0).reshape(m*d,1)
    dd = np.repeat(datasets,m,axis=1).reshape(m*d,1)
    df_temp_reshaped_data = np.concatenate((acc_data,cc,dd),axis=1)
    res = pd.DataFrame(data=df_temp_reshaped_data, index=range(d*m),
        columns=['accuracy','classifier_name','dataset_name'])
    res ['archive_name'] = 'UCR_TS_Archive_2015'
    return res

def generate_results_csv(output_file_name, root_dir, add_bake_off=True):
    res = pd.DataFrame(data = np.zeros((0,7),dtype=np.float), index=[],
        columns=['classifier_name','archive_name','dataset_name',
        'precision','accuracy','recall','duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir,archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0 :
                    curr_archive_name = curr_archive_name +'_itr_'+str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir+'/results/'+classifier_name+'/'\
                    +curr_archive_name+'/'+dataset_name+'/'+'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat( (res,df_metrics) ,axis=0,sort=False)

    res.to_csv(root_dir+output_file_name, index = False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy' : res.groupby(
            ['classifier_name','archive_name','dataset_name'])['accuracy'].mean()
        }).reset_index()

    if add_bake_off:
        res_bake_off = generate_results_csv_from_bake_off(root_dir=root_dir)
        res = add_results_from_bake_off(res,res_bake_off)

    return res

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val,y_pred_val)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin() 
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr == True:
        # print('row_best_model')
        # print(row_best_model)
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)
    # print('df_best_model')
    # print(df_best_model)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
    plot_epochs_metric(hist, output_directory+'epochs_loss.png')

    return df_metrics

def plot_pairwise(root_dir,classifier_name_1, classifier_name_2,
        res_df=None,title='', fig=None,color='green',label=None):
    if fig is None:  
        plt.figure()
    else: 
        plt.figure(fig)

    if res_df is None:
        res_df = generate_results_csv('results.csv',root_dir)
        res_df = add_themes(res_df)

    sorted_df = res_df.loc[(res_df['classifier_name']==classifier_name_1) | \
        (res_df['classifier_name']==classifier_name_2)].\
        sort_values(['classifier_name','archive_name','dataset_name'])
    # number of classifier we are comparing is 2 since pairwise 
    m = 2
    # get max nb of ready datasets 
    # count the number of tested datasets per classifier 
    df_counts = pd.DataFrame({'count': sorted_df.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets 
    max_nb_datasets = df_counts['count'].max()
    min_nb_datasets = df_counts['count'].min()
    # both classifiers should have finished 
    assert(max_nb_datasets==min_nb_datasets)

    data = np.array(sorted_df['accuracy']).reshape(m,max_nb_datasets).transpose()

    # concat the dataset name and the archive name to put them in the columns s
    sorted_df['archive_dataset_name'] = sorted_df['archive_name']+'__'+\
        sorted_df['dataset_name']
    # create the data frame containg the accuracies 
    df_data = pd.DataFrame(data=data, columns=np.sort([classifier_name_1,classifier_name_2]), 
        index = np.unique(sorted_df['archive_dataset_name']))

    # assertion 
    p1 = float(sorted_df.loc[(sorted_df['classifier_name']==classifier_name_1)&
        (sorted_df['dataset_name']=='Beef')]['accuracy'])
    p2 = float(df_data[classifier_name_1]['UCR_TS_Archive_2015__Beef'])
    assert(p1==p2)
    x = np.arange(start=0,stop=1,step=0.01)
    plt.xlim(xmax=1.02,xmin=0.0)
    plt.ylim(ymax=1.02,ymin=0.0)

    plt.scatter(x=df_data[classifier_name_1],y=df_data[classifier_name_2],color='blue')
                # c=sorted_df['theme_colors'])
    plt.xlabel('without data augmentation',fontsize='large')
    plt.ylabel('with data augmentation',fontsize='large')
    plt.plot(x,x,color='black')
    # plt.legend(loc='upper left')
    plt.title(title)

    uniq, counts = np.unique(df_data[classifier_name_1] < df_data[classifier_name_2], return_counts=True)
    print('Wins', counts[-1])

    uniq, counts = np.unique(df_data[classifier_name_1] == df_data[classifier_name_2], return_counts=True)
    print('Draws', counts[-1])


    uniq, counts = np.unique(df_data[classifier_name_1] > df_data[classifier_name_2], return_counts=True)
    print('Losses', counts[-1])

    p_value = wilcoxon(df_data[classifier_name_1], df_data[classifier_name_2], zero_method='pratt')[1]
    print(p_value)
    
    plt.savefig(root_dir+'/'+classifier_name_1+'-'+classifier_name_2+'_'+title+'.pdf'
        ,bbox_inches='tight')

def add_themes(df_perf):
    for dataset_name in DATASET_NAMES:
        df_perf.loc[df_perf['dataset_name']==dataset_name,'theme']= \
            utils.constants.dataset_types[dataset_name]
        df_perf.loc[df_perf['dataset_name'] == dataset_name, 'theme_colors'] = \
            utils.constants.themes_colors[utils.constants.dataset_types[dataset_name]]
    return df_perf

def plot_avg_acc_with_respect_to(df_transfer_mean,root_dir,root_dir_out,archive_name,title,
        datasets_dict=None,fig = 340):

    if datasets_dict is None: 
        datasets_dict = read_all_datasets(root_dir,archive_name)
    
    datasets_t = df_transfer_mean.index
    sizes = np.array([len(datasets_dict[d][0]) for d in datasets_t])
    sizes=100*sizes/sizes.sum()
    plt.figure(fig)
    plt.scatter(np.array(sizes),df_transfer_mean.values)
    plt.ylabel('avg_acc',fontsize='large')
    plt.xlabel('train_size',fontsize='large')
    plt.title(title)
    plt.savefig(root_dir_out+title+'.pdf',bbox_inches='tight')
    return datasets_dict,sizes 

def plot_compare_curves(root_dir,root_dir_transfer_learning,classifier_name,
        archive_name):
    from scipy.interpolate import spline
    metric = 'loss'
    # loop through the combination of datasets 
    for dataset_name_1 in DATASET_NAMES: 
        for dataset_name_2 in DATASET_NAMES: 
            # skip if transfering from the same dataset 
            if dataset_name_1 == dataset_name_2: 
                continue
            # read the original history
            df_history_or = pd.read_csv(root_dir+'results/'+classifier_name+
                '/'+archive_name+'/'+dataset_name_2+'/history.csv')
            # read the history data frame for transfer  
            df_history_tf = pd.read_csv(root_dir_transfer_learning+'/'+
                    dataset_name_1+'/'+dataset_name_2+'/history.csv') 

            max_epoch = df_history_tf.shape[0]
            smoothness = 300
            plt.figure()
            plt.title('Source: '+dataset_name_1+' - Target: '+dataset_name_2,fontsize='x-large')
            plt.ylabel('model\'s '+metric,fontsize='large')
            plt.xlabel('epoch',fontsize='large')
            plt.ylim(ymax=3.5)
            # plot orignal train
            y = df_history_or[metric].iloc[0:max_epoch]
            x = y.keys()
            tenth = int(len(x)/1) 
            if tenth%2 == 1:
                tenth = tenth+1
            w = tenth+1
            y = y.rolling(window=w,center=False,min_periods=1).mean()
            # linear interpolate to smooth 
            x_new = np.linspace(x.min(),x.max(),smoothness)
            y_new = spline(x,y,x_new)
            y = pd.Series(data=y_new,index=x_new)
            plt.plot(y,label='train_original',
                color=(255/255,160/255,14/255)) ## original train 
            # plot orignal test
            y = df_history_or['val_'+metric].iloc[0:max_epoch]
            x = y.keys()
            tenth = int(len(x)/1) 
            if tenth%2 == 1:
                tenth = tenth+1
            w = tenth+1
            y = y.rolling(window=w,center=False,min_periods=1).mean()
            # linear interpolate to smooth 
            x_new = np.linspace(x.min(),x.max(),smoothness)
            y_new = spline(x,y,x_new)
            y = pd.Series(data=y_new,index=x_new)
            plt.plot(y,label='test_original',
                color=(210/255,0/255,0/255)) ## original test 
            # plot transfer train
            y = df_history_tf[metric].iloc[0:max_epoch]
            x = y.keys()
            tenth = int(len(x)/1) 
            if tenth%2 == 1:
                tenth = tenth+1
            w = tenth+1
            y = y.rolling(window=w,center=False,min_periods=1).mean()
            # linear interpolate to smooth 
            x_new = np.linspace(x.min(),x.max(),smoothness)
            y_new = spline(x,y,x_new)
            y = pd.Series(data=y_new,index=x_new)
            plt.plot(y,label='train_transfer',
                color=(181/255,87/255,181/255)) # transfer train 
            # plot transfer test
            y = df_history_tf['val_'+metric].iloc[0:max_epoch]
            x = y.keys()
            tenth = int(len(x)/1) 
            if tenth%2 == 1:
                tenth = tenth+1
            w = tenth+1
            y = y.rolling(window=w,center=False,min_periods=1).mean()
            # linear interpolate to smooth 
            x_new = np.linspace(x.min(),x.max(),smoothness)
            y_new = spline(x,y,x_new)
            y = pd.Series(data=y_new,index=x_new)
            plt.plot(y,label='test_transfer',
                color=(27/255,32/255,101/255)) # transfer test
            plt.legend(loc='best')

            plt.savefig(root_dir_transfer_learning+'/'+
                    dataset_name_1+'/'+dataset_name_2+'/transfer.png',bbox_inches='tight')
            plt.close()

    exit()

def read_tf_matrix(df_perf,archive_name,root_dir,root_dir_out,root_dir_transfer_learning,classifier_name,
        datasets,max_nb_datasets,method_val = False, acc_improvement=True): 
    # init the transfer learning matrix 
    df_transfer = pd.DataFrame(data=np.zeros((max_nb_datasets,max_nb_datasets),dtype=np.float64),
        index= datasets, columns=datasets)
    # loop through the combination of datasets 
    for dataset_name_1 in datasets: 
        for dataset_name_2 in datasets: 
            # skip if transfering from the same dataset 
            if dataset_name_1 == dataset_name_2: 
                cell_value = 0 
                if method_val == 'train_val':
                    # read the original train accuracy csv 
                    df_best_model = pd.read_csv(root_dir+'results/'+classifier_name+'/'+archive_name
                        +'/'+dataset_name_2+'/df_best_model.csv')
                        # we assume the accuracy of the original model when transfer from dataset_1 to dataset_1
                    cell_value = df_best_model['best_model_train_acc'][0]
            else:
                try:
                    # read the metrics 
                    df_metrics = pd.read_csv(root_dir_transfer_learning+'/'+
                        dataset_name_1+'/'+dataset_name_2+'/df_metrics.csv')
                    if method_val =='cross_val':
                        # just save the validation accuracy 
                        cell_value = df_metrics['accuracy_val'][0]
                    elif method_val == 'train_val': 
                        # just save the best training accuracy
                        # read the necessary csv 
                        df_metrics = pd.read_csv(root_dir_transfer_learning+'/'+
                            dataset_name_1+'/'+dataset_name_2+'/df_best_model.csv')
                        cell_value = df_metrics['best_model_train_acc'][0]
                    elif method_val == 'no_val':
                        # get the accuracy when using transfer learning 
                        accuracy_transfer = df_metrics['accuracy'][0]
                        # get the accuracy without transfer learning (trian from scratch)
                        accuracy_original = np.array(df_perf.loc[(df_perf['classifier_name']==classifier_name) &\
                            (df_perf['dataset_name']==dataset_name_2)]['accuracy'])[0]
                        # calculate the percentage of accuracy imporvement 
                        cell_value = accuracy_transfer
                        if acc_improvement == True: 
                            cell_value = cell_value - accuracy_original
                            cell_value = 100*(cell_value)/accuracy_original

                except Exception as e:
                    print('Exception for:',dataset_name_1,dataset_name_2)
                    print(e)
                    exit()

            df_transfer[dataset_name_2][dataset_name_1] = cell_value

    if acc_improvement == False: 
        df_transfer.to_csv(root_dir_out+'df_transfer_acc.csv')

    if method_val == 'no_val' and acc_improvement == True:
        df_transfer.to_csv(root_dir_out+'df_transfer.csv')

    return df_transfer

def plot_avg_dba_vs_nn(df_perf,root_dir,df_transfer,datasets,root_dir_out,title):
    from scipy.interpolate import spline
    nn_max = 84 

    # read dba similarity calculations 
    df_similar_datasets = pd.read_csv(root_dir+'dba-python/similar_datasets.csv',index_col=0)
    
    idx = np.array(range(1,nn_max+1))

    outdir = root_dir_out+'datasets-plots/'
    create_directory(outdir)

    for d in datasets:

        acc_curve = [] 
        source_datasets = df_similar_datasets.loc[d].values
        for nn in idx: 
            avg_acc = df_transfer[d][source_datasets[nn-1]]
            acc_curve.append(avg_acc)

        y = np.array(acc_curve)
        x = idx 
        y = pd.Series(data=y,index = x)
        # similar to sp2m to get the window length for the moving average
        tenth = int(len(x)/1) 
        if tenth%2 == 1:
            tenth = tenth+1
        w = tenth+1

        y_new=y.rolling(window=w,center=False,min_periods=1).mean()
        # smoothness 
        smoothness = 300 
        # linear interpolate to smooth 
        x_new = np.linspace(x.min(),x.max(),smoothness)
        y_new = spline(x,y_new,x_new)
        X = x_new
        Y = y_new 
        acc_curve = pd.Series(data=Y,index = X)

        plt.figure()
        plt.xlabel('Rank_in_nearest_neighbor',fontsize='large')
        plt.ylabel('Accuracy',fontsize='large')

        # original segment 
        yy = df_perf[(df_perf['classifier_name']=='fcn') & \
            (df_perf['dataset_name']==d)]['accuracy'].values[0]
        yy = np.repeat(yy,len(idx))
        ss = pd.Series(data=yy,index=x)

        interval = 0.1
        # ymax 
        ymax = max(acc_curve.max(),ss.max())
        ymax = ymax + interval
        ymax = min(ymax,1.0)
        # ymin 
        ymin = min(acc_curve.min(),ss.min())
        ymin = ymin - interval
        ymin = max(ymin,0.0)

        plt.ylim(ymax=ymax,ymin=ymin)
        plt.plot(acc_curve,label='after_transfer',color='blue')

        # plot original 
        
        plt.plot(ss,color='red',label='original')
        plt.legend(loc='best')
        
        plt.title(d)
        plt.savefig(outdir+d+'.pdf',bbox_inches='tight')
        plt.close()

def plot_avg_dba_vs_rndm(classifier_name,df_perf,root_dir,df_transfer,datasets,root_dir_out,title,fig=7162): 
    nb_itr = 1000
    nb_neighbors = 1

    # read dba similarity calculations 
    df_similar_datasets = pd.read_csv(root_dir+'dba-python/similar_datasets.csv',index_col=0)
    avg_acc_dba = [] 
    for d in datasets:
        avg_acc = 0
        # for itr in range(nb_itr):
        source_datasets = df_similar_datasets.loc[d].values
        # idx_rndm = np.random.permutation(len(source_datasets))[:nb_neighbors]
        avg_acc += df_transfer[d][source_datasets[:nb_neighbors]].mean()
        # avg_acc_dba.append(avg_acc/nb_itr)
        avg_acc_dba.append(avg_acc)
    avg_acc_dba = np.array(avg_acc_dba)

    # # average accuracies inside dataset themes
    # # the name is just dba but it averages accuracies inside theme 
    # avg_acc_dba = [] 
    # for d in datasets:
    #     # get the theme 
    #     theme = df_perf.loc[df_perf['dataset_name']==d]['theme'].iloc[0]
    #     avg_acc = 0
    #     for itr in range(nb_itr):
    #         # get 
    #         source_datasets = df_perf.loc[df_perf['theme']==theme]['dataset_name'].values
    #         idx_rndm = np.random.permutation(len(source_datasets))[:nb_neighbors]
    #         avg_acc += df_transfer[d][source_datasets[idx_rndm]].mean()
    #     avg_acc_dba.append(avg_acc/nb_itr)
    # avg_acc_dba = np.array(avg_acc_dba)

    # do the same but with random dataset as sources for the transfer 
    avg_acc_rndm = []
    for d in datasets:
        avg_acc = 0
        for itr in range(nb_itr):
            source_datasets = datasets 
            idx_rndm = np.random.permutation(len(source_datasets))[:nb_neighbors]
            avg_acc += df_transfer.loc[source_datasets[idx_rndm]][d].mean()
        avg_acc_rndm.append(avg_acc/nb_itr)
    avg_acc_rndm = np.array(avg_acc_rndm)

    # plot bar 
    plt.figure(fig)

    avg_acc_rndm_series = pd.Series(data=avg_acc_rndm,index = datasets)
    avg_dba_series = pd.Series(data=avg_acc_dba,index = datasets)

    p_value = wilcoxon(avg_dba_series,avg_acc_rndm_series, zero_method = 'pratt')[1]
    print(p_value)
    
    # print('Wine - dtw ',avg_dba_series['Wine'])
    # print('Wine - rnd ',avg_acc_rndm_series['Wine'])
    # print('DiatomSizeReduction - dtw ',avg_dba_series['DiatomSizeReduction'])
    # print('DiatomSizeReduction - rnd ',avg_acc_rndm_series['DiatomSizeReduction'])
    # exit()

    # idx = [0,1,2,3]
    # print(avg_acc_rndm_series[datasets[idx]])
    # print(avg_dba_series[datasets[idx]])

    # dba better than rnd   
    idx = np.array(np.where(
        np.abs(avg_dba_series[datasets].values)
        > np.abs(avg_acc_rndm_series[datasets].values)))[0]
    p_dba = plt.bar(idx,avg_dba_series[datasets[idx]].values,color='red')
    p_rnd = plt.bar(idx,avg_acc_rndm_series[datasets[idx]].values,color='blue')

    # dba equal to rnd 
    idx = np.array(np.where(
        np.abs(avg_dba_series[datasets].values)
        == np.abs(avg_acc_rndm_series[datasets].values)))[0]
    p_dba = plt.bar(idx,avg_dba_series[datasets[idx]].values,color='gray')
    p_rnd = plt.bar(idx,avg_acc_rndm_series[datasets[idx]].values,color='gray')

    # dba worst than rnd 
    idx = np.array(np.where(
        np.abs(avg_dba_series[datasets].values)
        < np.abs(avg_acc_rndm_series[datasets].values)))[0]
    p_rnd = plt.bar(idx,avg_acc_rndm_series[datasets[idx]].values,color='blue')
    p_dba = plt.bar(idx,avg_dba_series[datasets[idx]].values,color='red')

    # print losses wins and 
    print('DBA vs RND')
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        > avg_acc_rndm_series[datasets].values))[0]
    print('\t Wins',len(idx))
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        == avg_acc_rndm_series[datasets].values))[0]
    print('\t Draws',len(idx))
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        < avg_acc_rndm_series[datasets].values))[0]
    print('\t Losses',len(idx))

    
    plt.title(title)
    plt.ylabel('worst_acc_improvement')

    plt.legend((p_rnd[0],p_dba[0]),('random','dtw'),loc='best')


    plt.savefig(root_dir_out+'random-vs-dba.pdf',bbox_inches='tight')
    plt.close()

    # plot pairwise 

    x = np.arange(start=0,stop=1,step=0.01)
    plt.xlim(xmax=1.02,xmin=0.0)
    plt.ylim(ymax=1.02,ymin=0.0)

    plt.scatter(y=avg_acc_dba,x=avg_acc_rndm,color='blue')
    plt.ylabel('Similar source (DTW)',fontsize='large')
    plt.xlabel('Random source',fontsize='large')
    plt.plot(x,x,color='black')
    plt.legend(loc='best')
    plt.title('Accuracy with transfer learning')
    
    plt.savefig(root_dir_out+'dtw_vs_rnd.pdf',bbox_inches='tight')
    plt.close()


    print('RND',avg_acc_rndm.mean(),'std',avg_acc_rndm.std())
    print('DBA',avg_acc_dba.mean(),'std',avg_acc_dba.std())

    return avg_dba_series

def plot_acc_1NN_dtw_transfer_vs_no_transfer(root_dir,datasets,df_transfer,df_perf,
        classifier_name,root_dir_out):
    # read dba similarity calculations 
    df_similar_datasets = pd.read_csv(root_dir+'dba-python/similar_datasets.csv',index_col=0)
    avg_acc_dba = [] 
    for d in datasets:
        source_datasets = df_similar_datasets.loc[d].values
        avg_acc = df_transfer[d][source_datasets[0]]
        avg_acc_dba.append(avg_acc)
    avg_acc_dba = np.array(avg_acc_dba)

    avg_dba_series = pd.Series(data=avg_acc_dba,index = datasets)

    # plot pairwise angainst no transfer 
    plt.figure()
    x = np.arange(start=0,stop=1,step=0.01)
    plt.xlim(xmax=1.02,xmin=0.0)
    plt.ylim(ymax=1.02,ymin=0.0)
    plt.plot(x,x,color='black')
    
    no_transfer_acc = df_perf.loc[df_perf['classifier_name']==classifier_name]
    no_transfer_acc = pd.Series(data=no_transfer_acc['accuracy'].values, 
        index=no_transfer_acc['dataset_name'].values)

    plt.scatter(y=avg_acc_dba,x=no_transfer_acc[datasets],color='blue')

    plt.ylabel('Similar_source_trasnfer',fontsize='large')
    plt.xlabel('No_transfer',fontsize='large')
    plt.legend(loc='best')
    # plt.title('Accuracy with transfer learning (1-NN-DTW) vs no trasnfer learning')
    
    plt.savefig(root_dir_out+'dtw_trasnfer-vs-no_transfer-pairwise.pdf',bbox_inches='tight')
    plt.close()

    # print losses wins and 
    print('dtw_transfer vs no_transfer')
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        > no_transfer_acc[datasets].values))[0]
    print('\t Wins',len(idx))
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        == no_transfer_acc[datasets].values))[0]
    print('\t Draws',len(idx))
    idx = np.array(np.where(
        avg_dba_series[datasets].values
        < no_transfer_acc[datasets].values))[0]
    print('\t Losses',len(idx))

def visualize_transfer_learning(root_dir,transfer_learning_dir='fcn_2000_train_all', classifier_name ='fcn',
        should_reload=False, sort_by='accuracy',method_val = 'no_val'): 
    ###############################################################
    ## get the classifer performance without transfer-learning ####
    ###############################################################
    # directory of transfer learning results 
    root_dir_transfer_learning = root_dir+'transfer-learning-results/'+transfer_learning_dir+'/'
    # directory for the output of transfer learning visualization 
    root_dir_out = root_dir+transfer_learning_dir+'/'
    # create directory if not exists 
    create_directory(root_dir_out)
    # the archive name we want to work with 
    archive_name = 'UCR_TS_Archive_2015'
    # get the nb of datasets (should be 85 for UCR archive)
    max_nb_datasets = len(DATASET_NAMES)
    # read the results of dl-tsc without transfer learning  
    df_perf = generate_results_csv('results.csv',root_dir,add_bake_off=True)
    # save a copy of the original untouched dl-tsc results 
    original_perf = df_perf.copy()
    # add the themes 
    df_perf = add_themes(df_perf)

    # get the performance of the classifier we are applying transfer learning to 
    df_perf = df_perf.loc[(df_perf['classifier_name']==classifier_name)&\
        (df_perf['archive_name']==archive_name)].sort_values([sort_by])
    # get the list of the datasets - sorted by 'sort_by' parameter 
    datasets = df_perf['dataset_name'].values

    ####################################
    ### transfer learning performance ##
    ####################################

    # check if it should be recomputed
    if should_reload == True:
        df_transfer = read_tf_matrix(df_perf,archive_name,root_dir,root_dir_out,root_dir_transfer_learning,classifier_name,
            datasets,max_nb_datasets,method_val = 'no_val')
    else:
        df_transfer = pd.read_csv(root_dir_out+'df_transfer.csv',index_col=0)
        df_transfer.index = datasets

    title = transfer_learning_dir

    if method_val != 'no_val':
        df_transfer_cv = read_tf_matrix(df_perf,archive_name,root_dir,root_dir_out,root_dir_transfer_learning,classifier_name,
            datasets,max_nb_datasets,method_val = method_val)

        print('Choose best transfer based on cross validation split:')
        # get the source dataset with maximum performance over each target dataset's val set
        df_transfer_cv_idmax = df_transfer_cv.idxmax(axis=0)
        df_transfer_cv_idmax = df_transfer_cv_idmax.values
        # get the performance on the corresponding test set
        df_transfer_agg_data = np.zeros(max_nb_datasets,)
        for i in range(max_nb_datasets):
            df_transfer_agg_data[i] = df_transfer.loc[df_transfer_cv_idmax[i]][datasets[i]]
        df_transfer_agg = pd.Series(data=df_transfer_agg_data,index=datasets)
        new_classifier_name = 'fcn_max_cv'
        plot_pairwise_transfer_learning(root_dir_out,classifier_name,archive_name,
            df_perf,df_transfer_agg,2323,'red',title,new_classifier_name,agg='agg')
        print('#######')

    print('Plotting history epoch loss compare transfer vs no transfer')
    plot_compare_curves(root_dir,root_dir_transfer_learning,classifier_name,
        archive_name)
    print('#######')

    print('Average accuracy using DBA, random or themes')
    df_transfer_temp = read_tf_matrix(df_perf,archive_name,root_dir,root_dir_out,root_dir_transfer_learning,
        classifier_name,datasets,max_nb_datasets,method_val = 'no_val',acc_improvement=False)
    avg_dba_series = plot_avg_dba_vs_rndm(classifier_name,df_perf,root_dir,df_transfer_temp,datasets,root_dir_out,title)
    plot_avg_dba_vs_nn(df_perf,root_dir,df_transfer_temp,datasets,root_dir_out,title)
    plot_acc_1NN_dtw_transfer_vs_no_transfer(root_dir,datasets,df_transfer_temp,
        df_perf,classifier_name,root_dir_out)
    df_transfer_temp= None 
    print('#######')
    # exit()

    print('Avg accuracy imporvements when transferring from these datasets:')
    df_transfer_mean = df_transfer.mean(axis=1) 
    df_transfer_std = df_transfer.std(axis=1) 
    id_max = df_transfer_mean.idxmax()
    id_max_from = id_max
    id_min = df_transfer_mean.idxmin()
    print('Max',id_max,'avg',df_transfer_mean[id_max],'std',df_transfer_std[id_max])
    print('Min',id_min,'avg',df_transfer_mean[id_min],'std',df_transfer_std[id_min])
    print('#######')

    print('Plotting avg acc (when transferring from) with respect to size')
    datasets_dict,sizes=plot_avg_acc_with_respect_to(df_transfer_mean,root_dir,root_dir_out,archive_name,
        'avg_acc-vs-train_size-transfer_from',datasets_dict=None,fig=99)
    print('#######')
    
    print('Avg accuracy imporvements when transferring to these datasets:')
    df_transfer_mean = df_transfer.mean(axis=0) 
    df_transfer_std = df_transfer.std(axis=0) 
    id_max = df_transfer_mean.idxmax()
    id_min = df_transfer_mean.idxmin()
    print('Max',id_max,'avg',df_transfer_mean[id_max],'std',df_transfer_std[id_max])
    print('Min',id_min,'avg',df_transfer_mean[id_min],'std',df_transfer_std[id_min])
    print('#######')

    print('Plotting avg acc (when transferring to) with respect to size')
    plot_avg_acc_with_respect_to(df_transfer_mean,root_dir,root_dir_out,archive_name,
        'avg_acc-vs-train_size-transfer_to',datasets_dict=datasets_dict,fig=99)
    print('#######')

    print('Plotting avg_acc_transfer_from vs avg_acc_transfer_to')
    plot_avg_acc(root_dir_out,datasets,sizes,df_transfer,fig=5)
    print('#######')

    new_classifier_name = classifier_name+'_transfer'
    color_ = (252/255,222/255,0/255)
    title_ = 'Aggregated accuracies with or without transfer'
    plot_pairwise_transfer_learning(root_dir_out,classifier_name,archive_name,
        df_perf,df_transfer,1,'blue',title_,new_classifier_name,agg='maximum')
    plot_pairwise_transfer_learning(root_dir_out,classifier_name,archive_name,
        df_perf,df_transfer,1,'red',title_,new_classifier_name,agg='minimum')
    plot_pairwise_transfer_learning(root_dir_out,classifier_name,archive_name,
        df_perf,df_transfer,1,color_,title_,new_classifier_name,agg='median')
    print('#######')

    print('Drawing the matrix pdf')
    fig, ax = plt.subplots(figsize=(30,30))
    plot_sns = sns.heatmap(ax=ax,data=df_transfer, cmap='RdBu', 
        xticklabels=True, yticklabels=True,vmin=-100,vmax=100)
    plot_sns.collections[0].colorbar.ax.tick_params(labelsize=30)
    plot_sns.collections[0].colorbar.set_label('% of accuracy variation after transfer',size=30)
    figure_name = root_dir_out+'heat_map.pdf'
    plot_sns.get_figure().savefig(figure_name,bbox_inches='tight')
    print('#######')

def plot_avg_acc(root_dir_out,datasets,sizes,df_transfer,fig=5):
    df_transfer_mean_transfer_from = df_transfer.mean(axis=1) 
    df_transfer_mean_transfer_to = df_transfer.mean(axis=0)

    # x is transfer from 
    # y is transfer to 
    plt.figure(fig)
    # remove the biggest one for visual 
    sizes [sizes.argmax()] = 7.5
    # color map 
    c= np.array(sizes,dtype=np.float64)
    plt.scatter(df_transfer_mean_transfer_from,df_transfer_mean_transfer_to,
        c=c,cmap='brg',edgecolors='none') 

    plt.xlabel('avg_acc_transfer_from',fontsize='large')
    plt.ylabel('avg_acc_transfer_to',fontsize='large')
    plt.legend()
    plt.colorbar(ticks=[]).set_label('size')
    plt.savefig(root_dir_out+'avg_acc_from-vs-avg_acc_to.pdf',bbox_inches='tight')

def plot_pairwise_transfer_learning(root_dir,classifier_name,archive_name,
        df_perf,df_transfer,fig,color,title,new_classifier_name,agg='max',already_calculated=False):
    """
    """
    print('Drwaing',agg,' accuracy plot with transfer learning\
        vs without transfer learning')

    if agg=='maximum':
        df_transfer_agg = df_transfer.max(axis=0)
    elif agg == 'minimum': 
        df_transfer_agg = df_transfer.min(axis=0)
    elif agg=='agg': 
        # already aggregated 
        df_transfer_agg = df_transfer
    elif agg=='median': 
        df_transfer_agg = df_transfer.median(axis=0)
    else: 
        df_transfer_agg = df_transfer.loc[agg]

    df_perf= concat_to_df_perf(df_perf,df_perf,df_transfer_agg,classifier_name,
        new_classifier_name,archive_name,print_win_losses=True,
        already_calculated = already_calculated)

    plot_pairwise(root_dir,classifier_name,new_classifier_name,
        res_df=df_perf, title=title, fig=fig,color=color,label=agg)

def concat_to_df_perf(df_perf,df_to_concat,df_transfer_agg,classifier_name,new_classifier_name,
        archive_name,print_win_losses=False,already_calculated = False):
    original_acc = df_perf.loc[df_perf['classifier_name']==classifier_name]['accuracy'].values 
    if already_calculated == False:
        # calculate the accuracy using the percentrage already computed 
        df_transfer_agg[:] = (df_transfer_agg.values*original_acc)/100 + original_acc

    # win loss draw
    if print_win_losses == True:
        print(classifier_name,'vs',new_classifier_name) 
        uniq , counts = np.unique(original_acc<df_transfer_agg.values,return_counts=True)
        print('\tWin:',counts[-1])
        uniq , counts = np.unique(original_acc>df_transfer_agg.values,return_counts=True)
        print('\tLosses:',counts[-1])
        uniq , counts = np.unique(original_acc==df_transfer_agg.values,return_counts=True)
        print('\tDraw:',counts[-1])

    # create a dataframe 
    df_perf_transfer = pd.DataFrame({'dataset_name':df_transfer_agg.index,
        'accuracy':df_transfer_agg.values})
    # add the neccessary attributes 
    df_perf_transfer['classifier_name']=new_classifier_name
    df_perf_transfer['archive_name'] = archive_name
    # concat with the original classifier's results 
    return pd.concat([df_to_concat,df_perf_transfer],sort=False)