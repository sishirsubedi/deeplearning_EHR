import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from matplotlib_venn import venn2

df= pd.read_csv("~/ghub/Data/medical_notes.csv")
df.head(2)
df.shape
df=df.iloc[:,[0,2]]
df.CATEGORY.unique()
df= df.loc[df.CATEGORY=='Discharge summary',:]
df.shape
len(df.SUBJECT_ID.unique())
df.isnull().values.any()
df.to_csv("~/ghub/Data/medical_records2.csv",index=False)

df= pd.read_csv("~/ghub/Data/medical_records2.csv")
df.head(2)
df['SUBJECT_ID'].value_counts().hist()


df2 = df.set_index(['SUBJECT_ID', df.groupby('SUBJECT_ID').cumcount()])['TEXT'].unstack().add_prefix('TEXT').reset_index()
df2.head(2)



#df2['TEXT']=df2.apply(lambda x:'%s_%s_%s_%s_%s' % (x['TEXT0'],x['TEXT1'],x['TEXT2'],x['TEXT3'],x['TEXT4']),axis=1)

df2['TEXT']=df2.apply(lambda x:'%s' % (x['TEXT0']),axis=1)

df2.head(1)
todrop_columns = list(range(1,df2.shape[1]-1))
df2.drop(df2.columns[todrop_columns], inplace=True, axis=1)


df_pdata = pd.read_csv("~/ghub/Data/patient_information.csv")
df_pdata.head(2)

df_final_pnotes = pd.merge(df2,df_pdata, on="SUBJECT_ID")
df_final_pnotes.to_csv("~/ghub/Data/Patient_notes_forNLP_small.csv",index=False)

#### finished preparing a single patient file for NLP ####
### analysis section ############

df = pd.read_csv("~/ghub/Data/Patient_notes_forNLP_small.csv")
df.shape
df.head(2)

patient = df.loc[ ( df.return_within_30days ==1) & (df.age>0)]
# patient = patient.iloc[random.sample(range(1790), 1000),0:2]
patient = patient.iloc[:,[0,1]]
patient.head(2)
patient.shape

normal_patient = df.loc[ ( df.times_of_admission ==1) & (df.age>0) & (df.died_in_first_admission ==0)]
normal_patient = normal_patient.iloc[random.sample(range(29452), 1794),0:2]
normal_patient.shape


all_documents = patient.TEXT.values

vectorizer = CountVectorizer(stop_words ='english')
vectorizer.fit_transform(all_documents)
print (vectorizer.vocabulary_)


smatrix = vectorizer.transform(all_documents)
smatrix.todense()

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(smatrix)

tf_idf_matrix = tfidf.transform(smatrix)
print (tf_idf_matrix.todense())

pd_tf_idf = pd.DataFrame(tf_idf_matrix.toarray())

pd_tf_idf.columns= [key for (key, value) in sorted(vectorizer.vocabulary_.items(), reverse=False)]

#analyze patient wise
for i in range(0,4):
    row = pd_tf_idf.iloc[i,:].values
    print("group----------")
    for j in range(0,10):
        print(pd_tf_idf.columns[np.argmax(row)])
        row[np.argmax(row)] =0

# analyze group
feature_sum = pd_tf_idf.sum(axis=0).values
row = feature_sum
patient_words = []
for i in range(0,50):
    print(pd_tf_idf.columns[np.argmax(row)])
    patient_words.append(pd_tf_idf.columns[np.argmax(row)])
    row[np.argmax(row)] =0


## run

all_documents = normal_patient.TEXT.values

vectorizer = CountVectorizer(stop_words ='english')
vectorizer.fit_transform(all_documents)
print (vectorizer.vocabulary_)

smatrix = vectorizer.transform(all_documents)
smatrix.todense()

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(smatrix)

tf_idf_matrix = tfidf.transform(smatrix)
print (tf_idf_matrix.todense())

pd_tf_idf = pd.DataFrame(tf_idf_matrix.toarray())

pd_tf_idf.columns= [key for (key, value) in sorted(vectorizer.vocabulary_.items(), reverse=False)]

#analyze patient wise
for i in range(0,4):
    row = pd_tf_idf.iloc[i,:].values
    print("group----------")
    for j in range(0,10):
        print(pd_tf_idf.columns[np.argmax(row)])
        row[np.argmax(row)] =0

# analyze group
feature_sum = pd_tf_idf.sum(axis=0).values
row = feature_sum
normal_words = []
for i in range(0,50):
    print(pd_tf_idf.columns[np.argmax(row)])
    normal_words.append(pd_tf_idf.columns[np.argmax(row)])
    row[np.argmax(row)] =0

patient = list(patient_words)
normal = list(normal_words)

venn2([set(patient), set(normal)])
plt.show()

print ("patient words ---")
for item in patient:
    if item not in normal:
        print (item)

print ("normal words ---")
for item in normal:
    if item not in patient:
        print (item)


combine = pd.concat([patient,normal_patient])
combine.head(2)
combine.shape

all_documents = combine.TEXT.values

vectorizer = CountVectorizer(stop_words ='english')
vectorizer.fit_transform(all_documents)
print (vectorizer.vocabulary_)


keywords = ['follow','moderate','home','normal','relax','severe','bleeding','infection','abnormal','critical']



smatrix = vectorizer.transform(all_documents)
smatrix.todense()

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(smatrix)

tf_idf_matrix = tfidf.transform(smatrix)
print (tf_idf_matrix.todense())

pd_tf_idf = pd.DataFrame(tf_idf_matrix.toarray())
pd_tf_idf.columns= [key for (key, value) in sorted(vectorizer.vocabulary_.items(), reverse=False)]

pd_tf_idf_keywords = pd_tf_idf[keywords]
pd_tf_idf_keywords.shape
pd_tf_idf_keywords.head(2)


pd_tf_idf_keywords = pd_tf_idf_keywords.reset_index(drop=True)
combine = combine.reset_index(drop=True)

pd_tf_idf_keywords['SUBJECT_ID'] = combine['SUBJECT_ID']
pd_tf_idf_keywords.head(2)
pd_tf_idf_keywords.to_csv("~/ghub/Data/nlp_output.csv",index=False)

df_icustays= pd.read_csv("~/ghub/Data/ICUSTAYS.csv")
df_icustays.head(2)
df_icustays = df_icustays.iloc[:,[1,3]]

pd_tf_idf_keywords_withicuids = pd.merge(pd_tf_idf_keywords,df_icustays,on='SUBJECT_ID')
pd_tf_idf_keywords_withicuids.head(2)
pd_tf_idf_keywords_withicuids.to_csv("~/ghub/Data/nlp_output.csv",index=False)