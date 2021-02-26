# Healthcare AI Experiments

## Data
Download the healthcare respiratory rate (RR) and heart rate (HR) datasets here: *http://timeseriesregression.org*

The *ts* data files can be converted using the awesome method *load_from_tsfile_to_dataframe* in https://github.com/ChangWeiTan/TS-Extrinsic-Regression/blob/master/utils/data_loader.py


Once downloaded, split the datasets in train,valid and test set according to a 70%/15%/15% ratio and save them
in **RR** or **HR** subdirectory of a **data** directory as numpy arrays, i.e. files
*trainx.npy*, *trainy.npy*, *validx.npy*, *validy.npy*, *testx.npy*, *testy.npy*
## Usage
After downloading the dataset, the training gets started by simply running:
```
python health_care_task.py [args]
```

Options:
- name : name of the dataset: HR or RR
- nhid : hidden size of recurrent net
- epochs : number of epochs
- batch : batch size
- lr : learning rate
- nlayers : number of layers
- dt : step size dt of UnICORNN
- alpha : y controle parameter alpha of UnICORNN
