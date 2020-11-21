# Time series classification for predictive maintenance on event logs

This is the companion repository of the "Time series classification for predictive maintenance on event logs" paper.

# Abstract

Time series classification (TSC) gained a lot of attention in the past decade and number of methods for representing and classifying time series have been proposed.
Nowadays, methods based on convolutional networks and ensemble techniques represent the state of the art for time series classification. Techniques transforming time series to image or text also provide reliable ways to extract meaningful features or representations of time series. We compare the state-of-the-art representation and classification methods on a specific application, that is predictive maintenance from sequences of event logs. The contributions of this paper are twofold: introducing a new data set for predictive maintenance on automatic teller machines (ATMs) log data and comparing the performance of different representation methods for predicting the occurrence of a breakdown. The problem is difficult since unlike the classic case of predictive maintenance via signals from sensors, we have sequences of discrete event logs occurring at any time and the length of the sequences, corresponding to life cycles, varies a lot.

When using this repository or the ATM dataset, please cite:

** Link to paper **

## Required packages

The experiment were conducted with python 3.8, the following packages are required to run the script:

* numpy
* scikit-learn
* pyts
* matrixprofile
* sktime
* pandas

If you wish to run ResNet for images classification, you will also need Tensorflow 2.x.

## How to get the ATM dataset
The ATM dataset being a property of equensWorldline, you must first send an email to "intellectual-property-team-worldline@worldline.com" and "antoine.guillaume@equensworldline.com" to ask for authorization. 
The compressed archive weights around 50Mo for a total weight of 575Mo. The dictionary of event codes will be supplied at the same time.

## Parameters & Configuration

Configuration parameters are located at the beginning of CV_script, you MUST change the base_path to match the current directory of this project. Other parameters can be left as is to reproduce the results of the paper.
To change or check the algorithms parameters, they all are redefined in custom wrapper classes to avoid errors, if a parameter is not specified in the constructor, it is left as default.
The representations methods are defined inside utils.representations and the classifications methods inside utils.classifications.

## Usage

Extract the files of the dataset archive located in ~/datasets in the dataset folder

```bash
python CV_script.py
```
The bash script launch_cross_validation.sh can be used on linux systems to run this as a background task with nohup, you MUST change the path to python to your own in the script. The script can also be imported inside a jupyter-notebook environment.

It is recommended that you run this script on a machine with at least 10 CPU cores, so all cross validation steps for a pipeline can run at the same time.
Expected runtime is 7 to 8 hours with 10 cores. 

To obtain results from TS-CHIEF: CV_script in his default configuration will export data formatted for the TS-CHIEF java version available at https://github.com/dotnet54/TS-CHIEF. A jar executable is already packaged including some debugging to make it runnable. Once TS-CHIEF data is exported you can run it with the following script (for linux systems):
```bash
bash ts-chief_script.sh
```
If you changed the path to the data for TS-CHIEF, make sure to report this change in this script.

The runtime of this script is extremely long, one iteration take about 4 hours, with 40 iterations to make for all cross validation splits and data encodings. Outputted results can then be formatted the same way as other results with the python script:
```bash
python TSCHIEF_results_to_csv.py
```
## Note on using sktime-dl for InceptionTime and ResNet
Both InceptionTime and ResNet are left commented in the code, so you can run the other algorithms without a Tensorflow installation or a GPU without any impact.
Depending on your installation, you might run into errors while feeding tensorflow models in a cross validation pipeline from scikit-learn. Some of those issues can be fixed by making the wrapper for those models defined in utils.classifications inheriting the KerasClassifier wrapper from tensorflow.

To make those two algorithms part of the experiments, you have to uncomment both their declaration in utils.classifications and the associated pipeline in CV_script.

About InceptionTime : sktime-dl is the package dedicated for deep learning built by the sktime authors, still being in active development at time of writing, we add to make some modifications to the source code to be able to run InceptionTime.
From the latest version available on github we applied the following modification :

* Fix import error from sktime utils : In sktime_dl/utils/_data.py, replace :
```
from sktime.utils.data_container import tabularize, from_3d_numpy_to_nested (_data.py line 6)
```
by
```
from sktime.utils.data_container import tabularize, from_3d_numpy_to_nested (_data.py line 6)
```

* We also modified InceptionTime to use binary_crossentropy (change loss name and use sigmod layer with 1 neuron as an output) and weighted accuracy for early stopping. This is not mandatory but is more suited to our problem.

## Contributing

If any bug should occur, please open a issue so we can work on a fix !

## Citations
[1]: [Johann Faouzi and Hicham Janati, "pyts: A Python Package for Time Series Classification", Journal of Machine Learning Research 2020](https://pyts.readthedocs.io/)


[2]: [Loning, Markus and Bagnall, Anthony and Ganesh, Sajaysurya and Kazakov, Viktor and Lines, Jason and Kiraly, Franz J, "sktime: A Unified Interface for Machine Learning with Time Series", Workshop on Systems for ML at NeurIPS 2019}](https://www.sktime.org/en/latest/)


[3]: [The Scikit-learn development team, "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research 2011](https://scikit-learn.org/stable/)


[4]: [The Pandas development team, "Pandas: Data Structures for Statistical Computing in Python"](https://pandas.pydata.org/)


[5]: [The Numpy development team, "Array programming with NumPy", Nature 2020](https://numpy.org/)
 
## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)