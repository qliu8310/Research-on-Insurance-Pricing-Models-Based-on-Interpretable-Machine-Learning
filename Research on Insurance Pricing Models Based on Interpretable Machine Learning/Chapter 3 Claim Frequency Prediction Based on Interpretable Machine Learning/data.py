def set_random_seeds(seed_nr):
    tf.random.set_seed(seed_nr)
    np.random.seed(seed_nr)
    random.seed(seed_nr)
    os.environ['PYTHONHASHSEED']=str(seed_nr)

set_random_seeds(42)

# 创建15个随机种子，用于后续训练15个不同的模型（可能是集成学习或交叉验证）
random_seeds = np.random.randint(0, 1000000000, 15)
df_freq = pd.read_feather(f'{storage_path}/Data/freMTPL2freq.feather')
df_sev = pd.read_feather(f'{storage_path}/Data/freMTPL2sev.feather')
# 使用 feather 格式读取数据（比 CSV 更快且保留数据类型）
df_freq = pd.read_feather(f'{storage_path}/Data/freMTPL2freq.feather')# 频率数据（包含特征和索赔数）
df_sev = pd.read_feather(f'{storage_path}/Data/freMTPL2sev.feather') # 赔款金额数据（包含特征和赔款金额）

# drop the column ClaimNb from df_freq:
df_freq = df_freq.drop(columns=["ClaimNb"])
# convert the column "VehGas" into categorical:
df_freq["VehGas"] = df_freq["VehGas"].astype("category")
# create a temporary dataframe with the column IDpol and the number of claims per policy:
temp_df_ClaimNb_from_sev_df = pd.DataFrame(df_sev["IDpol"].value_counts()).reset_index()
temp_df_ClaimNb_from_sev_df.columns = ["IDpol", "ClaimNb_from_sev_df"]
# we merge the two dataframes so that we have the column "ClaimNb_from_sev_df" in df_freq:
df_freq = pd.merge(df_freq, temp_df_ClaimNb_from_sev_df, on='IDpol', how='left')
df_freq["ClaimNb_from_sev_df"] = df_freq["ClaimNb_from_sev_df"].fillna(0)
# rename the column "ClaimNb_from_sev_df" to "ClaimNb":
df_freq = df_freq.rename(columns={"ClaimNb_from_sev_df":"ClaimNb"})
# replace all nan values of numerical columns in the dataframe with 0:
for col in df_freq.select_dtypes(include=['number']).columns:
    df_freq[col] = df_freq[col].fillna(0)
# restrict the dataframe to those raws that have a ClaimNb smaller or equal to 5:
df_freq = df_freq[df_freq["ClaimNb"]<=5]
df_freq = df_freq[(df_freq['VehAge'] >= 0) & (df_freq['VehAge'] <= 25) &
                  (df_freq['DrivAge'] >= 18) & (df_freq['DrivAge'] <= 80)]
# if exposure is bigger then 1 set it to 1:
df_freq.loc[df_freq["Exposure"]>1,"Exposure"] = 1
# reordering the categories of the column VehBrand to "B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14":
df_freq["VehBrand"] = df_freq["VehBrand"].cat.reorder_categories(["B1","B2","B3","B4","B5","B6","B10","B11","B12","B13","B14"])

# sort the dataframe by IDpol:
df_freq = df_freq.sort_values(by=["IDpol"])

ids_in_learn = list(np.genfromtxt(f"{storage_path}/Data/learn_split_IDpols_2.txt").astype(int))
ids_in_test = list(df_freq[~df_freq["IDpol"].isin(ids_in_learn)]["IDpol"].astype(int))

bool_in_learn = df_freq['IDpol'].isin(ids_in_learn) # be careful if the dataset is not sorted by IDpol
bool_in_test = df_freq['IDpol'].isin(ids_in_test) # be careful if the dataset is not sorted by IDpol

display(f"The learning data L contains so many instances: {len(ids_in_learn)}")
display(f"The test data T contains so many instances: {len(ids_in_test)}")

freq_learn = df_freq[bool_in_learn]['ClaimNb'].sum()/df_freq[bool_in_learn]['Exposure'].sum()
freq_test = df_freq[bool_in_test]['ClaimNb'].sum()/df_freq[bool_in_test]['Exposure'].sum()
display(f"Test the resulting portfolio freq (w.r.t Exposure) in learn df: {freq_learn: .2%}")
display(f"Test the resulting portfolio freq (w.r.t Exposure) in test df: {freq_test: .2%}")

# create 15 new train and validation split with sklearn:
train_val_split = {}
for run_index in range(15):
  temp_learn_train, temp_learn_val = sk.model_selection.train_test_split(df_freq[bool_in_learn][['IDpol']],
                                                                        test_size=0.1,
                                                                        random_state=random_seeds[run_index])
  train_val_split[f"learn_train_{run_index}"] = df_freq['IDpol'].isin(temp_learn_train['IDpol']) # be careful if the dataset is not sorted by IDpol
  train_val_split[f"learn_val_{run_index}"]  = df_freq['IDpol'].isin(temp_learn_val['IDpol']) # be careful if the dataset is not sorted by IDpol

print("Example train/validation split freq: ")
freq_learn_train = df_freq[train_val_split[f"learn_train_{run_index}"]]['ClaimNb'
                          ].sum()/df_freq[train_val_split[f"learn_train_{run_index}"]]['Exposure'].sum()
freq_learn_val = df_freq[train_val_split[f"learn_val_{run_index}"]]['ClaimNb'
                        ].sum()/df_freq[train_val_split[f"learn_val_{run_index}"]]['Exposure'].sum()

display(f"Test the resulting portfolio freq (w.r.t Exposure) in learn df: {freq_learn: .2%}")
display(f"Test the resulting portfolio freq (w.r.t Exposure) in learn-train df: {freq_learn_train: .2%}")
display(f"Test the resulting portfolio freq (w.r.t Exposure) in learn-val df: {freq_learn_val: .2%}")
del temp_learn_train, temp_learn_val

# create a new train and test split with sklearn:
temp_1, temp_lean_train_dummy = sk.model_selection.train_test_split(df_freq[train_val_split[f"learn_train_{run_index}"]][['IDpol']],
                                                                    test_size=0.01,
                                                                    random_state=random_seeds[0]+1)
bool_in_learn_train_dummy = df_freq['IDpol'].isin(temp_lean_train_dummy['IDpol']) # be careful if the dataset is not sorted by IDpol

freq_learn_train_dummy = df_freq[bool_in_learn_train_dummy]['ClaimNb'].sum()/df_freq[bool_in_learn_train_dummy]['Exposure'].sum()

display(f"Test the resulting portfolio freq (w.r.t Exposure) in learn train dummy df: {freq_learn_train_dummy: .2%}")
del temp_1, temp_lean_train_dummy

df_freq_prep_nn = df_freq.copy()


# change VehGas to binary:
df_freq_prep_nn["VehGas"] = df_freq_prep_nn["VehGas"].map({"Diesel":1,"Regular":0}).astype(int)

nr_col = [ "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehGas", "Density"]
cat_col = ["Area", "VehBrand", "Region"]

# Note: StandardScaler : = (x-mean)/standard_deviation
# Since it is good practice we are training the standardscaler (mean and standard_deviation) only the training data and apply it on the hole dataset (including the test data)
prep_standardscaler = StandardScaler()
prep_standardscaler.fit(df_freq_prep_nn[bool_in_learn][nr_col])


df_freq_prep_nn[nr_col] = prep_standardscaler.transform(df_freq_prep_nn[nr_col])
# add the dummy columns to the df_freq_prep_nn dataframe:
df_freq_prep_nn = pd.concat([df_freq_prep_nn.drop(columns=cat_col),
                             pd.get_dummies(df_freq_prep_nn[cat_col], columns=cat_col, drop_first=False).astype(int)
                             ], axis=1)
# add back the for the categorical columns that have been dropped above:
df_freq_prep_nn[list(map(lambda item: "Cat_" + item, cat_col))] = df_freq[cat_col]
cat_col = list(map(lambda item: "Cat_" + item, cat_col))

# Note we are not creating here every train and validation split dataset but instead create those when fitting the model.
# So that we are not polluting the RAM (notebooks have no garbage collector).



cat_encoder_all = {}
for col in ["Area", "VehBrand", "Region"]:
    cat_encoder = {}
    unique_cat = df_freq.dtypes[col].categories.to_list()
    for i in range(len(unique_cat)):
        cat_encoder[unique_cat[i]] = i
    cat_encoder_all[col]=cat_encoder # we save the encoder dict incase we will need it later to back transform the results.
    df_freq_prep_nn[f"NN_EMB_{col}"] = df_freq[col].map(cat_encoder_all[col]).astype(int)


# Create Datasets for OHE FNN:
# ------------------------
col_x_fnn_ohe = nr_col + [col for col in df_freq_prep_nn.columns if col.startswith('VehBrand_') or col.startswith('Region_')]
def create_ffn_ohe_data(bool_list, exposure_name="Exposure", response_name="ClaimNb"):
    X_nn_ohe = np.array(df_freq_prep_nn[bool_list][col_x_fnn_ohe].values)
    exposure = np.array(df_freq[bool_list][exposure_name])
    y_true= np.array(df_freq[bool_list][response_name])
    return [X_nn_ohe, exposure], y_true


# Create Datasets for cat embedding FNN:
# ------------------------
def create_ffn_cat_emb_data(bool_list, exposure_name="Exposure", response_name="ClaimNb"):
    X_nn_just_nr = np.array(df_freq_prep_nn[bool_list][nr_col].values)
    Input_EMB_Area = np.array(df_freq_prep_nn[bool_list]["NN_EMB_Area"].values)
    Input_EMB_VehBrand = np.array(df_freq_prep_nn[bool_list]["NN_EMB_VehBrand"].values)
    Input_EMB_Region = np.array(df_freq_prep_nn[bool_list]["NN_EMB_Region"].values)
    exposure = np.array(df_freq_prep_nn[bool_list][exposure_name])
    y_true = np.array(df_freq_prep_nn[bool_list][response_name])
    return [X_nn_just_nr, Input_EMB_Area, Input_EMB_VehBrand, Input_EMB_Region, exposure], y_true

