import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

features = pd.read_csv("~/ghub/Data/final_features.csv")
features.head(2)

df_nlp = pd.read_csv("~/ghub/Data/nlp_output.csv")
df_nlp.head(2)

nlp_features = features['features'].values

df_train= pd.read_csv("~/ghub/Data/train_data.csv")
df_train.head(2)
df_train.shape
X_train = df_train.iloc[:,0:df_train.shape[1]-1]
y_train = df_train.iloc[:,df_train.shape[1]-1]

X_train_nlp = pd.merge(X_train,df_nlp, on='icustay_id',how='left')
X_train_nlp.columns
X_train_nlp.shape
X_train_nlp = X_train_nlp.iloc[:,1:X_train_nlp.shape[1]-1]
X_train_nlp.head(1)
X_train = X_train_nlp.loc[:,nlp_features]
X_train.columns
X_train.shape


df_test= pd.read_csv("~/ghub/Data/test_data.csv")
df_test.head(2)
df_test.shape
X_test = df_test.iloc[:,0:df_test.shape[1]-1]
y_test = df_test.iloc[:,df_test.shape[1]-1]


X_test_nlp = pd.merge(X_test,df_nlp, on='icustay_id',how='left')
X_test_nlp.columns
X_test_nlp = X_test_nlp.iloc[:,1:X_test_nlp.shape[1]-1]
X_test_nlp.head(1)
X_test = X_test_nlp.loc[:,nlp_features]

X_test.fillna(0,inplace=True)
X_train.fillna(0,inplace=True)

### get name of features

##### feature analysis
lab = pd.read_csv("~/ghub/Data/D_LABITEMS.csv")
lab.head(2)
lab = lab.iloc[:,[1,2]]

chart = pd.read_csv("~/ghub/Data/D_ITEMS.csv")
chart.head(2)
chart = chart.iloc[:,[1,2]]

ids = pd.concat([lab,chart],axis=0)
ids.shape
ids.head(2)

temp = features['features'][10:20].copy()
features['features'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

nlp_features_name =[]
for f in features['features'].values:
    if f != '':
        #print (f)
        print(ids.loc[ids.ITEMID ==int(f),:].values[0][1])
        nlp_features_name.append(ids.loc[ids.ITEMID ==int(f),:].values[0][1])
    else:
        nlp_features_name.append(f)

nlp_features_name[10:20] = temp

# nlp_features_name = np.unique(nlp_features_name)

X_train.columns = nlp_features_name
X_test.columns = nlp_features_name

top10 = ['pH','Urea Nitrogen', 'Red Blood Cells',  'Eye Opening', 'Heart Rate', 'Respiratory Rate',
       'Chloride (serum)', 'Potassium (serum)', 'NBP Mean', 'Alanine Aminotransferase (ALT)']


top10index =[]
for i in top10:
    top10index.append(list(X_train.columns).index(i))

fig = plt.figure()
for i in range(len(top10index)):
    ax = plt.subplot(2, 5, i+1 )
    sns.distplot(X_train.iloc[:,top10index[i]],hist=False)
    sns.distplot(X_test.iloc[:,top10index[i]],hist=False)
plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
