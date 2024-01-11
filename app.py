import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns 
import pickle 

#import model
knn = pickle.load(open('cirrosis.pkl','rb'))

#load dataset
data = pd.read_csv('/content/drive/MyDrive/CirhossisDataset.csv')
def get_age(age):
  return int(age/360)

data.Age = data.Age.apply(get_age)

si = SimpleImputer(missing_values=np.nan,strategy="mean")
si.fit(data.iloc[:,10:-1])

data.iloc[:,10:-1] = si.transform(data.iloc[:,10:-1])



st.title('Aplikasi mendeteksi penyakit Liver')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Chirrosis Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Random Forest','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Pasien Dengan penyakit Liver</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('*Input Dataframe*')
    st.write(data)
    st.write('---')
    st.header('*Profiling Report*')
    st_profile_report(pr)

lii = list(data["Status"])

status = []
for i in range(len(lii)):
    if lii[i]=="D":
        status.append(0)
    elif lii[i]=="C":
        status.append(1)
    else :
        status.append(2)

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values


#train test split
X = data.drop('Stage',axis=1)
y = data['Stage']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():

    Age = st.sidebar.number_input('Age: ',20 , 60)
    Sex  = st.sidebar.selectbox('Sex',('M','F'))
    Ascites = st.sidebar.selectbox('ascites',('Y','N'))
    Hepatomegaly = st.sidebar.selectbox('hepatomegaly', ('Y','N'))
    Spiders = st.sidebar.selectbox('presence of spiders', ('Y','N'))
    Edema = st.sidebar.selectbox('presence of edema ',('Y','N', 'S'))
    Bilirubin = st.sidebar.number_input('serum bilirubin in [mg/dl]: ',0.4, 22.0)
    Cholesterol = st.sidebar.number_input('serum cholesterol in [mg/dl]: ',0, 1775)
    Albumin = st.sidebar.number_input('albumin in [gm/dl]: ', 0.0, 4.0)
    Copper = st.sidebar.number_input('urine copper in [ug/day] ', 0,588)
    Alk_Phos = st.sidebar.number_input('alkaline phosphatase in [U/liter]: ',0 ,11552)
    SGOT = st.sidebar.number_input('SGOT in [U/ml]: ',0, 338)
    Tryglicerides = st.sidebar.number_input('triglicerides in [mg/dl]: ', 0,598)
    Platelets = st.sidebar.number_input('platelets per cubic [ml/1000]  ',0,721)
    Prothrombin = st.sidebar.number_input(' prothrombin time in seconds [s]: ',9.5, 18.0)
    Stage = st.sidebar.number_input('histologic stage of disease (1, 2, 3, or 4): ', 1,4)
    
    user_report_data = {
        'Umur':Age,
        'sex':Sex,
        'Ascites':Ascites,
        'Hepatomelagy':Hepatomegaly,
        'Presence of spiders':Spiders,
        'Presence of edema':Edema,
        'serum bilirubin in [mg/dl]':Bilirubin,
        'serum cholesterol in [mg/dl]':Cholesterol,
        'albumin in [gm/dl]':Albumin,
        'urine copper in [ug/day]':Copper,
        'alkaline phosphatase in [U/liter]':Alk_Phos,
        'SGOT in [U/ml]':SGOT,
        'triglicerides in [mg/dl]':Tryglicerides,
        'platelets per cubic [ml/1000]':Platelets,
        'prothrombin time in seconds [s]':Prothrombin,
        'stage':Stage
        
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)


user_result = knn.predict(user_data)
knn_score = accuracy_score(y_test,knn.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Pasien ini aman'
else:
    output ='Pasien ini terkena penyakit Liver'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(knn_score*100)+'%')