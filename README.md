# Transfer learning for time series classification
This is the companion repository for [our paper](https://ieeexplore.ieee.org/document/8621990) titled "Transfer learning for time series classification" accepted as a regular paper at [IEEE International Conference on Big Data 2018](http://cci.drexel.edu/bigdata/bigdata2018/index.html) also available on [ArXiv](https://arxiv.org/abs/1811.01533). 

## Architecture
![architecture fcn](https://github.com/hfawaz/bigdata18/blob/master/png/fcn-archi.png)

## Source code
The software is developed using Python 3.5. We trained the models on a cluster of more than 60 GPUs. You will need the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) to re-run the experiments of the paper. 

If you encouter problems with cython, you can re-generate the "c" files using the [build-cython.sh](https://github.com/hfawaz/bigdata18/blob/master/utils/build-cython.sh) script. 

To train the network from scratch launch: ```python3 main.py train_fcn_scratch```

To apply the transfer learning between each pair of datasets launch: ```python3 main.py transfer_learning```

To visualize the figures in the paper launch: ```python3 main.py visualize_transfer_learning```

To generate the inter-datasets similariy matrix launch: ```python3 main.py compare_datasets```

### Pre-trained and fine-tuned models
You can download from the companion [web page](http://germain-forestier.info/src/bigdata2018/) all pre-trained and fine-tuned models you would need to re-produce the experiments. 
Feel free to fine-tune on your own datasets !!! 

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/bigdata18/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [cython](https://cython.org/)

## Results
You can download [here](https://github.com/hfawaz/bigdata18/blob/master/results/df_transfer.csv) the accuracy variation matrix which corresponds to the raw results of the transfer matrix in the paper.

You can download [here](https://github.com/hfawaz/bigdata18/blob/master/results/df_transfer_acc.csv) the raw results for the accuracy matrix instead of the variation.

You can download [here](https://github.com/hfawaz/bigdata18/blob/master/results/similar_datasets.csv) the result of the applying nearest neighbor algorithm on the inter-datasets similarity matrix. You will find for each dataset in the archive, the 84 most similar datasets.
The steps for computing the similarity matrix are presented in Algorithm 1 in our paper.

### Accuracy variation matrix
![acc-matrix](https://github.com/hfawaz/bigdata18/blob/master/png/acc-matrix.png)
### Generalization with and without the transfer learning
50words - FISH              |  FordA - wafer | Adiac - ShapesAll
:-------------------------:|:-------------------------:|:-------------------------:
![plot-50words-fish](https://github.com/hfawaz/bigdata18/blob/master/png/50words-fish.png)  |  ![plot-forda-wafer](https://github.com/hfawaz/bigdata18/blob/master/png/forda-wafer.png) | ![plot-adiac-shapesall](https://github.com/hfawaz/bigdata18/blob/master/png/adiac-shapesall.png)
### Model's accuracy with respect to the source dataset's similarity
Herring              |  BeetleFly | WormsTwoClass
:-------------------------:|:-------------------------:|:-------------------------:
![herring](https://github.com/hfawaz/bigdata18/blob/master/png/herring.png)  |  ![beetlefly](https://github.com/hfawaz/bigdata18/blob/master/png/beetlefly.png) | ![wormstwoclass](https://github.com/hfawaz/bigdata18/blob/master/png/wormstwoclass.png)

## Reference

If you re-use this work, please cite:

```
@InProceedings{IsmailFawaz2018transfer,
  Title                    = {Transfer learning for time series classification},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  booktitle                = {IEEE International Conference on Big Data},
  pages                    = {1367-1376}, 
  Year                     = {2018}
}
```

## Acknowledgement

The  authors  would  like  to  thank  NVIDIA  Corporation  for the GPU Grant and the Mésocentre of Strasbourg for providing access to the GPU cluster.
