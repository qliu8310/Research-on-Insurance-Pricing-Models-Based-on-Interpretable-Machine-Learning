# Copy the dataframe df_freq:
df_freq_glm = df_freq.copy()
# Area:
# is already numerical (due to the mapping above)
# VehPower:
temp_dict_change_VehPower={}
for i,v in enumerate(sorted(df_freq["VehPower"].unique())):
    if v <9:
        temp_dict_change_VehPower[v]=i+1
    else:
        temp_dict_change_VehPower[v]=6
df_freq_glm["VehPower"] = df_freq["VehPower"].map(temp_dict_change_VehPower).astype('category')
# VehAge:
# note: this part is different from the one in these papers:
# * 2018 Noll Case Study: "French Motor Third-Party Liability Claims"
# * 2019 Schelldorfer Paper: "Nesting Classical Actuarial Models into Neural Networks"
# * 2020 Wüthrich Paper: "From Generalized Linear Models to Neural Networks, and Back"
bins = [0, 6, 13, float('inf')]
labels = ['[0, 6)', '[6, 13)', '[13, ∞)']
df_freq_glm["VehAge"] = pd.cut(df_freq["VehAge"],bins=bins, labels=labels, right=False).astype('category')
# DrivAge:
bins = [18, 21, 26, 31, 41, 51, 71, float('inf')]
labels = ['[18, 21)', '[21, 26)', '[26, 31)', '[31, 41)', '[41, 51)', '[51, 71)', '[71, ∞)']
df_freq_glm["DrivAge"] = pd.cut(df_freq["DrivAge"],bins=bins, labels=labels, right=False).astype('category')
df_freq_glm["DrivAge_Nr"] = df_freq["DrivAge"]
# BonusMalus:
df_freq_glm.loc[df_freq_glm["BonusMalus"] >= 150, "BonusMalus"] = 150
# VehBrand:
# is already categorical (due to the reordering above)
# VehGas:
# is already categorical (due to the cast above)
# Density:
df_freq_glm["Density"] = np.log(df_freq_glm["Density"])
# Region:
# is already categorical

# check if we have the same number of features that we need for the glms as in the paper:
'''
print("Check if we have the same number of features that we need for the glms as in the paper")
print("------------")
test_dim_feature_space = 0
for col in df_freq_glm.select_dtypes(include=[int,float]).columns.drop(["IDpol","ClaimNb","Exposure","DrivAge_Nr"]):
    display(f"Dimensions for feature space of {col}: 1")
    test_dim_feature_space+=1
for col in df_freq_glm.select_dtypes(include=['category']).columns:
    display(f"Dimensions for feature space of {col}: {len(df_freq_glm[col].cat.categories)-1}")
    test_dim_feature_space=test_dim_feature_space+len(df_freq_glm[col].cat.categories)-1
display(f"Total dimensions for feature space: {test_dim_feature_space}")
'''

# Dummy encode all categorical variable for GLM1:
X_glm1 = pd.get_dummies(df_freq_glm, columns=df_freq_glm.select_dtypes(include=['category']).columns,drop_first=True).drop(columns=["IDpol","ClaimNb","Exposure","DrivAge_Nr"])
X_glm1_learn = X_glm1[bool_in_learn]
X_glm1_test = X_glm1[bool_in_test]

# Create the new DrivAge (power and log) columns for GLM2:
columns_to_drop = [col for col in X_glm1.columns if col.startswith('DrivAge_')]
X_glm2 = X_glm1.drop(columns=columns_to_drop)
X_glm2["DrivAge_1"] = df_freq_glm["DrivAge_Nr"]
X_glm2["DrivAge_2"] = df_freq_glm["DrivAge_Nr"]**2
X_glm2["DrivAge_3"] = df_freq_glm["DrivAge_Nr"]**3
X_glm2["DrivAge_4"] = df_freq_glm["DrivAge_Nr"]**4
X_glm2["DrivAge_log"] = np.log(df_freq_glm["DrivAge_Nr"])
X_glm2_learn = X_glm2[bool_in_learn].reset_index(drop=True)
means_DrivAge_learn = X_glm2_learn[[col for col in X_glm2_learn.columns if col.startswith('DrivAge_')]].mean()
for col in X_glm2_learn.columns:
    if col.startswith('DrivAge_'):
        X_glm2[col] = np.array(X_glm2[col]/means_DrivAge_learn[col])
X_glm2_learn = X_glm2[bool_in_learn].reset_index(drop=True)
X_glm2_test = X_glm2[bool_in_test].reset_index(drop=True)

# Adding interaction columns to the data frame for GLM3:
# one has to be careful here since the dataframes before are reindexed:
X_glm3 = X_glm2.copy()
X_glm3["DrivAge_1_x_BonusMalus"] = list(df_freq_glm["BonusMalus"]*df_freq_glm["DrivAge_Nr"])
X_glm3["DrivAge_2_x_BonusMalus"] = list(df_freq_glm["BonusMalus"]*df_freq_glm["DrivAge_Nr"]**2)
X_glm3_learn = X_glm2_learn.copy()
X_glm3_learn["DrivAge_1_x_BonusMalus"] = list(df_freq_glm[bool_in_learn]["BonusMalus"]*df_freq_glm[bool_in_learn]["DrivAge_Nr"])
X_glm3_learn["DrivAge_2_x_BonusMalus"] = list(df_freq_glm[bool_in_learn]["BonusMalus"]*df_freq_glm[bool_in_learn]["DrivAge_Nr"]**2)
means_DrivAge_x_BonusMalus_learn = X_glm3_learn[["DrivAge_1_x_BonusMalus", "DrivAge_2_x_BonusMalus"]].mean()
for col in list(means_DrivAge_x_BonusMalus_learn.index):
    X_glm3[col] = np.array(X_glm3[col]/means_DrivAge_x_BonusMalus_learn[col])
X_glm3_learn = X_glm3[bool_in_learn].reset_index(drop=True)
X_glm3_test = X_glm3[bool_in_test].reset_index(drop=True)



import numpy as np
df_freq_prep_nn["RandU"] = np.random.uniform(-np.sqrt(12)/2, np.sqrt(12)/2, len(df_freq_prep_nn))
df_freq_prep_nn["RandN"] = np.random.normal(0, 1, len(df_freq_prep_nn))

# Recreating results GLM1:
# -------------------------
for run_index in range(15):
    start_time = time.time()
    poisson_glm1 = PoissonRegressor(alpha = 0,max_iter=1000, solver='newton-cholesky') # scikit-learn.org: alpha = 0 is equivalent to unpenalized GLMs
    poisson_glm1.fit(X_glm1_learn,y_true["train"]/exposure["train"],sample_weight=exposure["train"])
    end_time = time.time()
    execution_time_glm1 = end_time - start_time
    # Make predictions using the fitted model
    y_pred["train"]["GLM1"] = poisson_glm1.predict(X_glm1_learn)*exposure["train"]
    y_pred["test"]["GLM1"] = poisson_glm1.predict(X_glm1_test)*exposure["test"]
    # store the results in the results class:
    glm1_results = Results(model=f"GLM1 (run: {run_index})",
                            epochs=0,
                            run_time=execution_time_glm1,
                            nr_parameters=len(poisson_glm1.coef_)+len([poisson_glm1.intercept_]),
                            poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"]["GLM1"]),
                            poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"]["GLM1"]),
                            pred_avg_freq_train=y_pred["train"]["GLM1"].sum()/exposure["train"].sum(),
                            pred_avg_freq_test=y_pred["test"]["GLM1"].sum()/exposure["test"].sum())
    # store the results in the dataframe:
    store_results_in_df(glm1_results)


# Recreating results GLM2:
# -------------------------
for run_index in range(15):
    start_time = time.time()
    poisson_glm2 = PoissonRegressor(alpha = 0,max_iter=1000, solver='newton-cholesky') # scikit-learn.org: alpha = 0 is equivalent to unpenalized GLMs
    poisson_glm2.fit(X_glm2_learn,y_true["train"]/exposure["train"],sample_weight=exposure["train"])
    end_time = time.time()
    execution_time_glm2 = end_time - start_time
    # Make predictions using the fitted model
    y_pred["train"]["GLM2"] = poisson_glm2.predict(X_glm2_learn)*exposure["train"]
    y_pred["test"]["GLM2"] = poisson_glm2.predict(X_glm2_test)*exposure["test"]
    # store the results in the results class:
    glm2_results = Results(model=f"GLM2 (run: {run_index})",
                            epochs=0,
                            run_time=execution_time_glm2,
                            nr_parameters=len(poisson_glm2.coef_)+len([poisson_glm2.intercept_]),
                            poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"]["GLM2"]),
                            poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"]["GLM2"]),
                            pred_avg_freq_train=y_pred["train"]["GLM2"].sum()/exposure["train"].sum(),
                            pred_avg_freq_test=y_pred["test"]["GLM2"].sum()/exposure["test"].sum())
    # store the results in the dataframe:
    store_results_in_df(glm2_results)


# Recreating results GLM3:
# -------------------------
for run_index in range(15):
    start_time = time.time()
    poisson_glm3 = PoissonRegressor(alpha = 0,max_iter=1000, solver='newton-cholesky') # scikit-learn.org: alpha = 0 is equivalent to unpenalized GLMs
    poisson_glm3.fit(X_glm3_learn,y_true["train"]/exposure["train"],sample_weight=exposure["train"])
    end_time = time.time()
    execution_time_glm3 = end_time - start_time

    # Make predictions using the fitted model
    y_pred["train"]["GLM3"] = poisson_glm3.predict(X_glm3_learn)*exposure["train"]
    y_pred["test"]["GLM3"] = poisson_glm3.predict(X_glm3_test)*exposure["test"]

    # store the results in the results class:
    glm3_results = Results(model=f"GLM3 (run: {run_index})",
                            epochs=0,
                            run_time=execution_time_glm3,
                            nr_parameters=len(poisson_glm3.coef_)+len([poisson_glm3.intercept_]),
                            poisson_deviance_loss_train=poisson_deviance_loss(y_true["train"], y_pred["train"]["GLM3"]),
                            poisson_deviance_loss_test=poisson_deviance_loss(y_true["test"], y_pred["test"]["GLM3"]),
                            pred_avg_freq_train=y_pred["train"]["GLM3"].sum()/exposure["train"].sum(),
                            pred_avg_freq_test=y_pred["test"]["GLM3"].sum()/exposure["test"].sum())
    # store the results in the dataframe:
    store_results_in_df(glm3_results)
display(df_results)