import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist



df_admission =pd.read_csv("~/ghub/Data/patient_admissions.csv")
df_admission.head(2)
df_admission= df_admission.loc[:,["SUBJECT_ID","HADM_ID","ADMITTIME","DISCHTIME","DEATHTIME"]]

df_patient =pd.read_csv("~/ghub/Data/patients_data.csv")
df_patient.head(2)
df_patient = df_patient.loc[:,["SUBJECT_ID","DOB","DOD","EXPIRE_FLAG"]]

merged = pd.merge(df_admission,df_patient, on=["SUBJECT_ID"])

merged.head(10)

all_patients = merged.iloc[:,0].values


pinfo=[] # id, age, stay once or more , return within 30days, died, died in first stay,

pid =[]
for id in all_patients:

    if id in pid:
        continue
    else:
        pid.append(id)

        temp = []

        row = merged.loc[merged['SUBJECT_ID']==id]

        temp.append(id) # id

        first_admit = datetime.strptime(row['ADMITTIME'].values[0], '%Y-%m-%d %H:%M:%S')
        dob = datetime.strptime(row['DOB'].values[0], '%Y-%m-%d %H:%M:%S')
        age = int(abs((dob - first_admit).days)/365.0)

        if age>80: age=80

        temp.append(age)

        if len(row)==1:

            temp.append(1) # stay once
            temp.append(0)  # no return within 30

            first_admit = datetime.strptime(row['ADMITTIME'].values[0], '%Y-%m-%d %H:%M:%S')
            first_discharge = datetime.strptime(row['DISCHTIME'].values[0], '%Y-%m-%d %H:%M:%S')
            stay_day = abs((first_admit - first_discharge).days)

            if row['EXPIRE_FLAG'].values[0] ==1:

                temp.append(1) # died

                deathday = datetime.strptime(row['DOD'].values[0], '%Y-%m-%d %H:%M:%S')
                death_day_diff = same_day = abs((first_admit - deathday).days)

                if stay_day <= death_day_diff :
                    temp.append(0)
                else:
                    temp.append(1) # died in first stay
            else:
                temp.append(0)
                temp.append(0)
        else:
            temp.append(len(row)) # how many times admitted

            first_admit = datetime.strptime(row['ADMITTIME'].values[0], '%Y-%m-%d %H:%M:%S')
            first_discharge = datetime.strptime(row['DISCHTIME'].values[0], '%Y-%m-%d %H:%M:%S')

            second_admit = datetime.strptime(row['ADMITTIME'].values[1], '%Y-%m-%d %H:%M:%S')
            second_discharge = datetime.strptime(row['DISCHTIME'].values[1], '%Y-%m-%d %H:%M:%S')

            days = abs((second_admit - first_discharge).days)

            if days<31:
                temp.append(1) # return within 30 days
            else:
                temp.append(days)  # return within 30 days

            temp.append(0) # died
            temp.append(0) # died within 30 days
        pinfo.append(temp)

df_pinfo = pd.DataFrame(pinfo)
df_pinfo.columns = ["patient_id", "age","times_of_admission","return_within_30days","died","died_in_first_admission"]
df_pinfo.to_csv("patient_information.csv",index=False)



### clustering


df_all_patientdata= pd.read_csv("~/ghub/Data/df_final_ICUid.csv")
df_all_patientdata.head(2)
df_all_patientdata.shape
X = df_all_patientdata.iloc[:,1:(df_all_patientdata.shape[1]-1)]
X.shape


# k means determine k
distortions = []
K = [10,15,20,25,30]
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


