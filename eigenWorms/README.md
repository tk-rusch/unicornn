# EigenWorms

## Data
Download the eigenworms dataset here: *http://www.timeseriesclassification.com/description.php?Dataset=EigenWorms*

The *arff* data files can be converted using **sktime** (pip install sktime).

Once downloaded, split the dataset in train,valid and test set according to a 70%/15%/15% ratio and save them
in a **data** directory as numpy arrays, i.e. files
*trainx.npy*, *trainy.npy*, *validx.npy*, *validy.npy*, *testx.npy*, *testy.npy*
## Usage
After downloading the dataset, the training gets started by simply running:
```
python eigenworms_task.py [args]
```

Options:
- nhid : hidden size of recurrent net
- epochs : number of epochs
- batch : batch size
- lr : learning rate
- nlayers : number of layers
- dt : step size dt of UnICORNN
- alpha : y controle parameter alpha of UnICORNN
