# Create the dataframes needed for evaluation:
# --------------------
# NOTE: in the 2021 Gorishniy et al paper the batch size is different for the different Datasets
# but is not hyperparameter tuned. Bigger datasets they used a batch size of 1024 and
# for smaller datasets a batch size of (256/512).
batch_size = 1024
learn_data = df_to_tensor(df_freq_prep_nn[bool_in_learn], feature_cols=nr_col+cat_col, exposure="Exposure", target="ClaimNb", batch_size=batch_size)
test_data = df_to_tensor(df_freq_prep_nn[bool_in_test], feature_cols=nr_col+cat_col, exposure="Exposure", target="ClaimNb", batch_size=batch_size)

# NOTE we use at first just a fraction of the data to test the code:
learn_train_dummy_data = df_to_tensor(df_freq_prep_nn[bool_in_learn_train_dummy], feature_cols=nr_col+cat_col, exposure="Exposure", target="ClaimNb", batch_size=batch_size,
                                      dummy_data_for_build=True)

for run_index in range(15):
    # Create the dataframes needed for training:
    learn_train_data = df_to_tensor(df_freq_prep_nn[train_val_split[f"learn_train_{run_index}"]],
                                    feature_cols=nr_col+cat_col, exposure="Exposure", target="ClaimNb", batch_size=batch_size)
    learn_val_data = df_to_tensor(df_freq_prep_nn[train_val_split[f"learn_val_{run_index}"]],
                                  feature_cols=nr_col+cat_col, exposure="Exposure", target="ClaimNb", batch_size=batch_size)

    print(f"-------------------------------------------------")
    print(f"-------------------------------------------------")
    print(f"-------------------------------------------------")
    print(f"-------We are at Model: {str(run_index).zfill(2)}-----------------------")
    print(f"-------------------------------------------------")
    print(f"-------------------------------------------------")
    print(f"-------------------------------------------------")
    # Define FT-Transformer Models:
    # ----------------------
    # NOTE: we use here tensorflow/keras model subclasses (not the functional or sequential api)
    # NOTE: we use here instead of the .fit function a costum training loop

    # create the model:
    # ----------------------
    set_random_seeds(int(random_seeds[run_index]))

    FT_transformer = EnhActuar.Feature_Tokenizer_Transformer(
            emb_dim = 32, # NOTE: In the default setting for the 2021 Gorishniy paper they used emb_dim = 192 (but the parameter size would here go trough the roof, so we use something smaller)
            nr_features = nr_col,
            cat_features = cat_col,
            cat_vocabulary = cat_vocabulary,
            count_transformer_blocks = 3,
            attention_n_heads = 8,
            attention_dropout = 0.2,
            ffn_d_hidden = None, # NOTE: change to None if ReGLU should be used -> None uses default value (4/3*emb_dim), they write that they used 2*emb_dim if not ReGLU.
            ffn_activation_ReGLU = True, # NOTE: set True if ReGLU should be used
            ffn_dropout = 0.1,
            prenormalization = True,
            output_dim = 1,
            last_activation = 'exponential',
            exposure_name = "Exposure",
            seed_nr = int(random_seeds[run_index])
    )

    # See here regarding costum training loop: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

    # Instantiate an optimizer to train the model.
    # ----------------------
    # create an optimizer AdamW with learning rate 1e-4, weight decay 1e-5:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5)

    # Instantiate a loss function
    # ----------------------
    # we use our own loss function here
    # because it is not included in tensorflow in the same way (see section loss function for more details):
    loss_fn = Poisson_loss_for_tf_Wrapped()

    # Prepare the metrics.
    # ----------------------
    # we use a costume metric here (because it is not included in tensorflow in the same way):
    train_acc_metric = Poisson_Metric_for_tf()
    val_acc_metric = Poisson_Metric_for_tf()
    test_acc_metric = Poisson_Metric_for_tf()

    @tf.function
    def train_step(x, y):
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            y_pred = FT_transformer(x, training=True)["output"]  # prediction for this minibatch
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y, y_pred)
        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, FT_transformer.trainable_weights)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, FT_transformer.trainable_weights))
        # Update training metric.
        train_acc_metric.update_state(y, y_pred)
        return loss_value

    @tf.function
    def val_step(x, y):
        # Run the forward pass of the layer.
        # (note: training=False is needed because the layers have different behavior during training versus inference (e.g. Dropout))
        y_pred = FT_transformer(x, training=False)["output"]
        # Update val metrics
        val_acc_metric.update_state(y, y_pred)

    @tf.function
    def test_step(x, y):
        # Run the forward pass of the layer.
        # (note: training=False is needed because the layers have different behavior during training versus inference (e.g. Dropout))
        y_pred = FT_transformer(x, training=False)["output"]
        # Update val metrics
        test_acc_metric.update_state(y, y_pred)

    # model fitting:
    # ----------------------
    start_time = time.time()
    Val_Progress = helper.Easy_ProgressTracker(patience=15)
    epochs = 500

    for epoch in range(epochs):
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(learn_train_data):
            loss_value = train_step(x_batch_train, y_batch_train)
            helper.costume_progress_bar(f"Ensemble: {str(run_index).zfill(2)}/{14} / Epoch: {epoch} / Batch: {step} / Train-Loss (Batch): {round(float(loss_value),4)}",step,len(learn_train_data), 30)

        # Display metrics at the end of each epoch.
        print_train_loss = train_acc_metric.result()
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation at the end of each epoch.
        for x_batch_val, y_batch_val in learn_val_data:
            val_step(x_batch_val, y_batch_val)
        print_val_loss = val_acc_metric.result()
        val_acc_metric.reset_states()
        for x_batch_test, y_batch_test in test_data:
            test_step(x_batch_test, y_batch_test)
        print_test_loss = test_acc_metric.result()
        test_acc_metric.reset_states()

        Val_Progress(current_epoch=epoch, current_score = print_val_loss)

        print(f"\nEnsemble: {str(run_index).zfill(2)}/{14} / Epoch: {epoch} / Train-Loss: %.4f / Val-Loss: %.4f / Test-Loss: %.4f / Time taken: %s / ---- Currently Best Val-Epoch: %d" % (
            # str(run_index).zfill(2),
            float(print_train_loss),
            float(print_val_loss),
            float(print_test_loss),
            datetime.timedelta(seconds=int(time.time() - start_time)),
            Val_Progress.best_epoch
            ), end = " ")
        if Val_Progress.progress == True:
            print("<------- Best VAL Epoch so far")
        else:
            print("\r")


        # Callback: save best model / early stopping:
        # ----------------------
        earliest_epoch2save = 10
        if Val_Progress.progress and Val_Progress.current_epoch >= earliest_epoch2save:
            # FT_transformer.save(storage_path +'/Poisson_FT_transformer')
            FT_transformer.save_weights(f'{storage_path}/saved_models/Poisson_FT_transformer_{run_index}.weights.h5')
        if Val_Progress.patience_over:
            break

    # create some metrics after the loop
    best_epoch_FT_transformer = Val_Progress.best_epoch
    execution_time_FT_transformer = time.time() - start_time

    # load the best saved model and epochs_and_time from the pickle file:
    # ----------------------
    # FT_transformer = keras.models.load_model(save_path +'/Poisson_FT_transformer')
    FT_transformer.load_weights(f'{storage_path}/saved_models/Poisson_FT_transformer_{run_index}.weights.h5')


# Predict with the model:
y_pred["train"][f"FT_transformer"] = np.array([x for [x] in FT_transformer.predict(learn_data, batch_size=100000)["output"]])
y_pred["test"][f"FT_transformer"] = np.array([x for [x] in FT_transformer.predict(test_data, batch_size=100000)["output"]])

# Evaluate the model:
FT_transformer_results = Results(
    model=f"FT_transformer (run: {run_index})",
    epochs=best_epoch_FT_transformer,
    run_time=execution_time_FT_transformer,
    nr_parameters=[np.sum([np.prod(v.shape.as_list()) for v in FT_transformer.trainable_weights])],
    poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"][f"FT_transformer"]),
    poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"][f"FT_transformer"]),
    pred_avg_freq_train=y_pred["train"][f"FT_transformer"].sum() / exposure["train"].sum(),
    pred_avg_freq_test=y_pred["test"][f"FT_transformer"].sum() / exposure["test"].sum()
)

# Store the results in the dataframe:
store_results_in_df(FT_transformer_results)
display(df_results)