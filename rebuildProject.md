## Project Configurations
```
# Runtime params
#===================================
train: True # train new or existing model for each channel
predict: True # generate new predicts or, if False, use predictions stored locally
use_id: "2018-05-19_15.00.10"

# number of values to evaluate in each batch
batch_size: 70

# number of trailing batches to use in error calculation
window_size: 30

# Columns headers for output file
header: ["run_id", "chan_id", "spacecraft", "num_anoms", "anomaly_sequences", "class", "true_positives", 
        "false_positives", "false_negatives", "tp_sequences", "fp_sequences", "gaussian_p-value", "num_values",
        "normalized_error", "eval_time", "scores"]

# determines window size used in EWMA smoothing (percentage of total values for channel)
smoothing_perc: 0.05

# number of values surrounding an error that are brought into the sequence (promotes grouping on nearby sequences
error_buffer: 100

# LSTM parameters
# ==================================
loss_metric: 'mse'
optimizer: 'adam'
validation_split: 0.2
dropout: 0.3
lstm_batch_size: 64

# maximum number of epochs allowed (if early stopping criteria not met)
epochs: 35

# network architecture [<neurons in hidden layer>, <neurons in hidden layer>]
# Size of input layer not listed - dependent on evr modules and types included (see 'evr_modules' and 'erv_types' above)
layers: [80,80]

# Number of consequetive training iterations to allow without decreasing the val_loss by at least min_delta 
patience: 10
min_delta: 0.0003

# num previous timesteps provided to model to predict future values
l_s: 250

# number of steps ahead to predict
n_predictions: 10

# Error thresholding parameters
# ==================================

# minimum percent decrease between max errors in anomalous sequences (used for pruning)
p: 0.13
```

## Step by Step Guidance:
###1. load data one by one according to the row in the labeled_anomalies.csv
###2. for each set of data:
#### (1) Load Data

For example, for P-1:

```
# l_s = 250 (num previous timesteps provided to model to predict future values), 
# n_predictions = 10 (number of steps ahead to predict）
data = [] 
for i in range(len(arr) - config.l_s - config.n_predictions):
    data.append(arr[i:i + config.l_s + config.n_predictions])
```
    
Thus:    
train_shape: ```(2872, 25)```

X_train_shape: `````(2612, 250, 25)````` 

y_train_shape: ```(2612, 10)```

X_test_shape: ```(8245, 250, 25)``` 

y_test_shape: ```(8245, 10)```

#### (2) Train Model
```
def get_model(anom, X_train, y_train, logger, train=False):
    '''Train LSTM model according to specifications in config.yaml or load pre-trained model.

    Args:
        anom (dict): contains all anomaly information for a given input stream
        X_train (np array): numpy array of training inputs with dimensions [timesteps, l_s, input dimensions)
        y_train (np array): numpy array of training outputs corresponding to true values following each sequence
        logger (obj): logging object
        train (bool): If False, will attempt to load existing model from repo

    Returns:
        model (obj): Trained Keras LSTM model 
    '''

    if not train and os.path.exists(os.path.join("data", config.use_id, "models", anom["chan_id"] + ".h5")):
        logger.info("Loading pre-trained model")
        return load_model(os.path.join("data", config.use_id, "models", anom["chan_id"] + ".h5"))

    elif (not train and not os.path.exists(os.path.join("data", config.use_id, "models", anom["chan_id"] + ".h5"))) or train:
        
        if not train:
            logger.info("Training new model from scratch.")

        
        '''
        # Callbacks
        History(): Callback that records events into a History object. History attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        ==> Used for plotting
        
        EarlyStopping(): Stop training when a monitored quantity has stopped improving.
        
        monitor: quantity to be monitored.
        
        patience: number of epochs that produced the monitored quantity with no improvement after which training will be stopped.Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one.
        
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        
        verbose: int. 0: quiet, 1: update messages. (show the msg per epoch)
        '''
        cbs = [History(), EarlyStopping(monitor='val_loss', patience=config.patience, 
            min_delta=config.min_delta, verbose=0)]
        
        # 1. Define Network
        # Keras Sequential models
        model = Sequential()

        '''
        LSTM layer with 80 nodes
        
        Pass an input_shape argument (None,25) to the first layer. This is a shape tuple (a tuple of integers or None entries, where None indicates that any positive integer may be expected).
        # expected input data shape: (batch_size, timesteps, data_dim)
        # input_shape=(timesteps, data_dim) 
        # Here, timesteps = None => Any positive integer
        # 25 => data_dimension
        # returns a sequence of vectors of dimension 80
        
        '''
        model.add(LSTM(
            config.layers[0],
            input_shape=(None, X_train.shape[2]),
            return_sequences=True))
        model.add(Dropout(config.dropout))

        model.add(LSTM(
            config.layers[1],
            return_sequences=False))
        model.add(Dropout(config.dropout))

        # Ouputting layer
        model.add(Dense(
            config.n_predictions))
        model.add(Activation("linear"))

        # Compile Network
        '''
        # Compilation is an efficiency step. It transforms the simple sequence of layers that we defined into a highly efficient series of matrix transforms in a format intended to be executed on your GPU or CPU, depending on how Keras is configured.
        # Here,
        # loss_metric: 'mse'
        # optimizer: 'adam'
        '''
        model.compile(loss=config.loss_metric, optimizer=config.optimizer) 

        # Fit Network
        # A stateful recurrent model is one for which the internal states (memories) obtained after processing a batch of samples are reused as initial states for the samples of the next batch
        model.fit(X_train, y_train, batch_size=config.lstm_batch_size, epochs=config.epochs, 
            validation_split=config.validation_split, callbacks=cbs, verbose=True)
        
        model.save(os.path.join("data", anom['run_id'], "models", anom["chan_id"] + ".h5"))

        return model
```

#### (3) Predict in batches that simulate a spacecraft downlinking schedule
```
y_hat = models.predict_in_batches(y_test, X_test, model, anom)
```
Here, y_hat is the predicted values (E.g. for P-1, y_hat.shape: ```(8245, 10)```

#### (4) Get errors
```
e = err.get_errors(y_test, y_hat, anom, smoothed=False)
e_s = err.get_errors(y_test, y_hat, anom, smoothed=True)
```
Here, e is abs(predicted_y - actual_y)
e_s is the smoothed errors using ewma (Exponentially Weighted Moving Average)

#### (5) Process errors 
```E_seq, E_seq_scores = err.process_errors(y_test, y_hat, e_s, anom, logger)```

Within function ```process_errors```, we process errors by batch, for each batch:
##### 5.1 find epsilon (standard_deviation threshold)
Find the anomaly threshold that maximizes function representing tradeoff between a) number of anomalies
    and anomalous ranges and b) the reduction in mean and st dev if anomalous points are removed from errors
```
epsilon = find_epsilon(window_e_s, config.error_buffer) # error_buffer == 100
```
Note that we find sd_threshold via:
```for sd_threshold in np.arange(2.5, sd_lim, 0.5)```, where ```sd_lin = 12.0``` by default.

##### 5.2 compare to epsilon and epsilon_inv
```
# find sequences of anomalies greater than epsilon
E_seq, i_anom, non_anom_max = compare_to_epsilon(e_s, epsilon, len_y_test,
inter_range, chan_std, std, config.error_buffer, window, i_anom_full)

# find sequences of anomalies using inverted error values (lower than normal errors are also anomalous)
E_seq_inv, i_anom_inv, inv_non_anom_max = compare_to_epsilon(e_s_inv, epsilon_inv, 
len_y_test, inter_range, chan_std, std, config.error_buffer, window, i_anom_full)
```

##### 5.3 prune anomalies
Remove anomalies that don't meet minimum separation from the next closest anomaly or error value with ```prune_anoms```

Here, minimum percent decrease between max errors in anomalous sequences (used for pruning) 
```p : 0.13``` 

======= Batch Processing Completed ======
##### 5.4 group anomalous indices into continuous sequences
```
i_anom = sorted(list(set(i_anom)))
groups = [list(group) for group in mit.consecutive_groups(i_anom)]
E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
```

##### 5.5 calc anomaly scores based on max distance from epsilon for each sequence
```
anom_scores = []
for e_seq in E_seq:
    score = max([abs(e_s[x] - epsilon) / (np.mean(e_s) + np.std(e_s)) for x in range(e_seq[0], e_seq[1])])
    anom_scores.append(score)
```

#### 6 Evaluate results 
Compare identified anomalous sequences with labeled anomalous sequences
```
anom = err.evaluate_sequences(E_seq, anom)
```

