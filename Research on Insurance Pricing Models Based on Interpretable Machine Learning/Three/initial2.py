# Loss-function (for numpy arrays)
# ----------------------
def poisson_deviance_loss(y_true, y_pred):
    with np.errstate(divide='ignore'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            xlogy = np.where(y_true != 0, y_true * np.log(y_true / y_pred), 0)
            dev = 2 * (xlogy - y_true + y_pred)
    return dev.mean()

# Loss-Function
# ----------------------
# we use our own loss function here (because it is not included in tensorflow in the same way):
# normally here i would use the tf loss class (using the LossFunctionWrapper but this does not work on colab
# since there is no @keras_export() in the source code...):
# @keras.saving.register_keras_serializable(package="my_package", name="poisson_loss_for_tf")
@tf.function()
def poisson_loss_for_tf(y_true, y_pred, mean=True):
    """Computes the Poisson loss between y_true and y_pred.

    The Poisson loss is the mean of the elements of the `Tensor`
    `2 * (y_true * log(y_true / y_pred) - y_true + y_pred)`.

    Args:
        y_true: A tensor of true values with shape (batch_size,).
        y_pred: A tensor of predicted values with shape (batch_size,).

    Returns:
        The Poisson loss between y_true and y_pred.
   """
   # NOTE: this squeeze is not very professional :) but it does its job right now...
    ''' TODO: check if this commented squeeze is needed or not?
    if y_pred.shape != y_true.shape:
        if y_pred.ndim > y_true.ndim:
            y_pred = tf.squeeze(y_pred, [-1])
        elif y_pred.ndim < y_true.ndim:
                y_true = tf.squeeze(y_true, [-1])
    '''
    if y_pred.shape != y_true.shape:
        y_pred = tf.squeeze(y_pred, [-1])
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    loss = 2 * (y_true * tf.math.log((y_true + keras.backend.epsilon()) / (y_pred + keras.backend.epsilon())) - y_true + y_pred)
    if mean:
        return tf.reduce_mean(loss, axis=-1)
    else:
        return loss

# Loss Function Wrapper
# ----------------------
class Poisson_loss_for_tf_Wrapped:
    def __init__(self, y_true=None, y_pred=None, name="poisson_loss_for_tf"):
        self.name = name
        self.y_true = y_true
        self.y_pred = y_pred
    def __call__(self, y_true, y_pred):
        return poisson_loss_for_tf(y_true, y_pred)


# Loss Metrics.
# ----------------------
# See here: 
class Poisson_Metric_for_tf(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super(Poisson_Metric_for_tf, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_poisson_loss = poisson_loss_for_tf(y_true, y_pred,mean=False)
        sum_batch_poisson_loss = tf.reduce_sum(batch_poisson_loss)
        num_samples = tf.cast(tf.size(y_true), tf.float32)
        if sample_weight is not None:
            raise ValueError('Code for sample_weight is not jet implemented')
        self.total.assign_add(sum_batch_poisson_loss)
        self.count.assign_add(num_samples)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0)
        self.count.assign(0)


# note that the function tf.keras.losses.Poisson() is not the same as the poisson_deviance_loss function above.
# the function tf.keras.losses.Poisson() is the same as mean(y_pred - y_true * tf.math.log(y_pred + 1e-10)).


def df_to_tensor(df: pd.DataFrame, feature_cols: list, exposure: str=None, target: str=None, batch_size: int = 512, dummy_data_for_build=False):
    """
    transforms the pandas dataframe to a tensorflow dataset as input for the model

    Args:
        df (pd dataframe): the pandas dataframe that includes the features
        feature_cols (list): the list of feature columns that should be included in the model
        exposure (str): if the exposure is included it will be used a a separate input (if None it will be ignored)
        target (str): if the target is included it will be used in as a separate input (if None it will be ignored)
        batch_size (int): the batch size for the tensorflow dataset
        dummy_data_for_build (bool): build a dummy dataset for the model (only for building the model) that is not prefetched (default: False)

    Returns:
        tensorflow Dataset (Prefetched and Batched)
    """
    if exposure:
        feature_cols = feature_cols+[exposure]
    temp_dict = {k.lower(): np.array(v).reshape(-1, 1).astype(np.float32, copy=False)
                            if v.dtype in ["float64","float32","int64","int32"] else
                            np.array(v).reshape(-1, 1) for k, v in df[feature_cols].items()}
    if target:
        temp_input = (temp_dict, np.array(df[target]))
    else:
        temp_input = (temp_dict)
    tf_dataset = tf.data.Dataset.from_tensor_slices(temp_input) # create the tf dataset
    tf_dataset = tf_dataset.batch(batch_size) # for parallelizing the calc
    if dummy_data_for_build == False:
        tf_dataset = tf_dataset.prefetch(batch_size) # Prefetch the data for better performance (helps to overlaps the data preprocessing and model execution)
    return tf_dataset

cat_vocabulary = {}
for c in cat_col:
    cat_vocabulary[c] = df_freq_prep_nn.dtypes[c].categories.tolist()


# init hash tables for results
y_pred = {}
y_pred["train"]={}
y_pred["test"]={}

y_true = {}
y_true["train"] = np.array(df_freq[bool_in_learn]["ClaimNb"])
y_true["test"] = np.array(df_freq[bool_in_test]["ClaimNb"])

exposure = {}
exposure["train"] = np.array(df_freq[bool_in_learn]["Exposure"])
exposure["test"] = np.array(df_freq[bool_in_test]["Exposure"])

log_exposure = {}
log_exposure["train"] = np.array(np.log(df_freq[bool_in_learn]["Exposure"]))
log_exposure["test"] = np.array(np.log(df_freq[bool_in_test]["Exposure"]))

epochs_and_time = {}

df_results = pd.DataFrame(columns=["model",
                                   "epochs",
                                   "run_time",
                                   "# parameters",
                                   "poisson deviance loss: train",
                                   "poisson deviance loss: test",
                                   f"pred-avg-freq: train (obs = {freq_learn: .2%})",
                                   f"pred-avg-freq: test (obs = {freq_test: .2%})"])

# create a python data class to store the results:
@dataclass
class Results:
    model: str
    epochs: int = field(default=None)
    run_time: float = field(default=None)
    nr_parameters: int = field(default=None)
    poisson_deviance_loss_train: float = field(default=None)
    poisson_deviance_loss_test: float = field(default=None)
    pred_avg_freq_train: float = field(default=None)
    pred_avg_freq_test: float = field(default=None)

# create a function that stores the results in a dataframe not using append since dataframe object has no attribute append:
def store_results_in_df(results):
    global df_results
    global freq_learn
    global freq_test
    if len(df_results[df_results["model"]!=results.model])==0:
        df_results = pd.DataFrame({"model":results.model,
                                            "epochs":results.epochs,
                                            "run_time":results.run_time,
                                            "nr_parameters":results.nr_parameters,
                                            "loss_train":results.poisson_deviance_loss_train,
                                            "loss_test":results.poisson_deviance_loss_test,
                                            f"pred_avg_freq_train":results.pred_avg_freq_train,
                                            f"pred_avg_freq_test":results.pred_avg_freq_test},
                                  index=[0])
    else:
        df_results = pd.concat([df_results[df_results["model"]!=results.model],
                                pd.DataFrame({"model":results.model,
                                                "epochs":results.epochs,
                                                "run_time":results.run_time,
                                                "nr_parameters":results.nr_parameters,
                                                "loss_train":results.poisson_deviance_loss_train,
                                                "loss_test":results.poisson_deviance_loss_test,
                                                f"pred_avg_freq_train":results.pred_avg_freq_train,
                                                f"pred_avg_freq_test":results.pred_avg_freq_test},
                                             index=[0])
                                ], ignore_index=True).reset_index(drop=True)


def calc_avg_df(list_models):
    for i, model in enumerate(list_models):
        filtered_results = df_results[df_results['model'].str.startswith(model)]
        averages = pd.DataFrame(filtered_results.select_dtypes(include=['number']).mean()).T
        averages.insert(0, 'model', model)
        if i == 0:
            df_avg = averages
        else:
            df_avg = pd.concat([df_avg, averages], ignore_index=True)
    return df_avg


def calc_std_df(list_models):
    for i, model in enumerate(list_models):
        filtered_results = df_results[df_results['model'].str.startswith(model)]
        averages = pd.DataFrame(filtered_results.select_dtypes(include=['number']).std()).T
        averages.insert(0, 'model', model)
        if i == 0:
            df_std = averages
        else:
            df_std = pd.concat([df_std, averages], ignore_index=True)
    return df_std

