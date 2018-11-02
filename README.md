# Transfer learning for time series classification
This is the companion repository for our paper titled "Transfer learning for time series classification" accepted as a regular paper at [IEEE International Conference on Big Data 2018](http://cci.drexel.edu/bigdata/bigdata2018/index.html) also available on ArXiv [[ADD LINK TODO]]. 

## Architecture
![architecture fcn](https://github.com/hfawaz/bigdata18/blob/master/png/fcn-archi.png)

## Source code
The software is developed using Python 3.5. We trained the models on a cluster of more than 60 GPUs. You will need the [UCR archive](https://www.cs.ucr.edu/~eamonn/time_series_data/) to re-run the experiments of the paper. 
### Pre-trained and fine-tuned models
You can download from the companion [web page](http://germain-forestier.info/bigdata2018/) all pre-trained and fine-tuned models you would need to re-produce the experiments. 
Feel free to fine-tune on your own datasets !!! 

## Prerequisites
All python packages needed are listed in [pip-requirements.txt] [[TDO ADD LINK]] file and can be installed simply using the pip command. 

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)

## Results
The raw results can be found on our companion [web page](http://germain-forestier.info/bigdata2018/).
### Accuracy variation matrix
![acc-matrix](https://github.com/hfawaz/bigdata18/blob/master/png/acc-matrix.png)
### Genrealization with and without the transfer learning
50words - FISH              |  FordA - wafer | Adiac - ShapesAll
:-------------------------:|:-------------------------:|:-------------------------:
![plot-50words-fish](https://github.com/hfawaz/bigdata18/blob/master/png/plot-50words-fish.png)  |  ![plot-forda-wafer](https://github.com/hfawaz/bigdata18/blob/master/png/plot-forda-wafer.png) | ![plot-adiac-shapesall](https://github.com/hfawaz/bigdata18/blob/master/png/plot-adiac-shapesall.png)
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
  booktitle                = {IEEE International Conference On Big Data},
  Year                     = {2018}
}
```
