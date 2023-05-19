# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

import pandas as pd
import numpy as np
import os
import folium
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, MultiLabelBinarizer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multioutput import MultiOutputClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
import streamlit as st
import xgboost as xgb
import numpy_financial as npf
import shap
from streamlit_folium import folium_static
import io


st.set_page_config(layout="centered",page_title='Churn Analysis and Prediction')

st.title('Understanding the drivers of Churn in a Telecom Company')
st.markdown('By: [Karshni Mitra](https://www.linkedin.com/in/karshnimitra/) & [Yi-Hsueh Yang](https://www.linkedin.com/in/yi-hsueh-alex-yang/)')
st.caption('Data Source: https://data.world/mcc450/telecom-churn')

path = './data'

st.markdown(" ")
st.markdown("A quick overview of the data")
churn_df = pd.read_csv(path+'/telecom_customer_churn.csv')
st.write("Dimensions of the data:",churn_df.shape)
st.dataframe(churn_df.head().T)

buffer = io.StringIO()
churn_df.info(buf=buffer)
info_str = buffer.getvalue()

# Display the captured output in Streamlit
with st.expander("Raw Churn Data Information"):
    st.text(info_str)

# Get the captured output as a string
info_str = buffer.getvalue()


st.header('Exploratory Data Analysis')

# Disable max row constraint
alt.data_transformers.disable_max_rows()

st.subheader("Exploring churn characteristics")

a = alt.Chart(churn_df).mark_bar().encode(
    x='Customer Status',
    y='count(Customer ID)',
).properties(
    width=500,
    height=450,
    title='Churn Count'
)

a

st.markdown("**Figure 1:** This shows the number of customers in our dataset by their status of 'Churned','Joined' and 'Stayed'.")

a = alt.Chart(churn_df).mark_bar().encode(
    x='Gender',
    y=alt.X('count(Customer ID)', stack='normalize'),
    color='Customer Status',
    tooltip='count(Customer ID)',
).properties(
    width=500,
    height=450,
    title='Gender VS Customer Status (Normalized)'
)

st.altair_chart(a)

st.markdown("**Figure 2:** Customer status by gender. As we can see, it is evenly distributed. The number of males and females who churned, joined and stayed are equal. We also see (by hovering on the bar graphs to get the absolute values) that the dataset is evenly distributed, 50% are males and 50% are females.")

a = alt.Chart(churn_df).mark_bar().encode(
    x=alt.X('Age',bin=alt.Bin(maxbins=20)),
    y=alt.Y('count(Customer ID)'),
    color='Customer Status',
    tooltip='count(Customer ID)',
).properties(
    width=600,
    height=450,
    title='Age VS Customer Status'
)

a

st.markdown("**Figure 3:** Customer status binned by Age. As we can see, it is evenly distributed over most age groups.")

a = alt.Chart(churn_df).mark_bar().encode(
    x='Married',
    y=alt.X('count(Customer ID)', stack='normalize'),
    color='Customer Status',
    tooltip='count(Customer ID)',
).properties(
    width=500,
    height=450,
    title='Married VS Customer Status (Normalized)'
)

a

st.markdown("**Figure 4:** Customer status binned by Marriage Status. People who are not married tend to churn more than those who are.")

c1 = alt.Chart(churn_df).mark_bar().encode(
    x='Customer Status',
    y='count(Customer ID)',
    tooltip='count(Customer ID)',
    color='Offer'
).properties(
    width=400,
    title='Customer status by offers given'
)

c2 = alt.Chart(churn_df).mark_bar().encode(
    x='Offer',
    y=alt.X('count(Customer ID)', stack='normalize'),
    color='Customer Status',
    tooltip='count(Customer ID)'
).properties(
    width=400,
    title='Offers given to customers and the outcome'
)


c = alt.vconcat(c1,c2).resolve_scale(color='independent')

st.altair_chart(c, use_container_width=True)

st.markdown("**Figure 5:** From the above charts, we can see that Offer E is very unsuccessful with people who may churn, since majority of people who are offered E still tend to churn. We also observe that Offer A has a higher success rate at retaining customers.")

bar = alt.Chart(churn_df).mark_bar().encode(
    x='Tenure in Months:O',
    y=alt.X('count(Customer ID)'),
    color='Customer Status',
    tooltip='count(Customer ID)'
).properties(
    width=900,
    height=450,
    title='Customer Status and their tenure in months'
) 

bar

st.markdown('''**Figure 6:** This shows us the number of customers and their status as their tenure increases. As we can see, there is a high amount of churn within the first 12 months as compared to people who stay. 
The amount of churn is highest in the first month. This proportion decreases as time goes on, and becomes extremely small after 4 years. 
            \nAn interesting observation here is that, for people who have just joined, the label assigned is 'Joined' instead of 'Stayed'. 
            Implications are that maybe the company offers a 3 month trail period, and categorizes people who stary after trial as 'Stayed'. 
            During model training, we will treat these 2 labels as the same.''')

grouped_churn = churn_df.groupby(['Tenure in Months','Customer Status']).count()[['Customer ID']]
grouped_churn.reset_index(inplace=True)
grouped_churn = grouped_churn[grouped_churn['Customer Status'] == 'Churned']

grouped_total = churn_df.groupby('Tenure in Months').count()[['Customer ID']]
grouped_total.reset_index(inplace=True)

grouped = pd.merge(grouped_churn,grouped_total,on='Tenure in Months').drop('Customer Status',axis=1)
grouped['Churn Percent'] = grouped['Customer ID_x'] / grouped['Customer ID_y']

# + colab={"base_uri": "https://localhost:8080/", "height": 505} id="IPzGI4nrpumn" executionInfo={"status": "ok", "timestamp": 1683395405826, "user_tz": 240, "elapsed": 1490, "user": {"displayName": "Karshni Mitra", "userId": "11351333322129634740"}} outputId="16f0ac34-b20f-4cb1-dbdd-f1812d4b848d"
line = alt.Chart(grouped).mark_line(point=True).encode(
    x='Tenure in Months:O',
    y=alt.Y('Churn Percent:Q', axis=alt.Axis(format='%'), title='Churn Percent'),
    color=alt.value('red'),
    tooltip=alt.Tooltip(['Churn Percent:Q'], format='.2%',title='Churn'),
).properties(
    width=900,
    height=450,
    title='Churn Percent by month'
) 

st.altair_chart((bar + line).configure_point(size=60).resolve_scale(y='independent'))

st.markdown("**Figure 7:** Same as Figure 6. The line shows churn % at each month. There are random spikes, but we can clearly see the steady downward trend over time.") 

a = alt.Chart(churn_df).mark_bar().encode(
    x='Number of Referrals:N',
    y=alt.X('count(Customer ID)'),
    color='Customer Status',
    tooltip='count(Customer ID)',
).properties(
    width=600,
    height=500,
    title='Number of referrals and customer status'
)
a
st.markdown('''**Figure 8:** This chart shows us the proportion of people who churned/stayed, based on the referrals they gave. Generally as we would expect, people who refer more people tend to be loyal customers, with almost no churn after 5/6 referrals. 
\n
 This could help the company in reducing churn by introducing referral offers and programs.''')

a = alt.Chart(churn_df.dropna(subset=['Avg Monthly GB Download'])).mark_bar().encode(
    alt.X('Avg Monthly GB Download:Q',bin=alt.Bin(maxbins=10)),
    y=alt.X('count(Customer ID)'),
    tooltip='count(Customer ID)',
    color='Customer Status'
).properties(
    width=800,
    height=400,
    title='Avg Monthly Downloads and Customer status'
)

st.altair_chart(a)

st.markdown("**Figure 9:** This chart shows the proportion of people who churned based on their average monthly usage. People using above 40GB, generally churn significantly lesser than people using lesser than that.")

a=alt.Chart(churn_df).mark_bar().encode(
    x='Internet Type',
    y=alt.X('count(Customer ID)'),
    tooltip='count(Customer ID)',
    color='Customer Status',
).properties(
    width=550,
    height=450,
    title='Internet type of Customers with different statuses'
)

st.altair_chart(a)

st.markdown("**Figure 10:** The chart above shows the proportion of people who churned based on the type of internet they have. Clearly a high proportion of people using fiber optic cable tend to churn. Let's dig a little deeper as to why.")

hover_selection = alt.selection_single(
    clear='mouseout',
    fields=['Churn Reason'],
    name='hover_selection',
    on='mouseover',
)

a = alt.Chart(churn_df[churn_df['Internet Type'] == 'Fiber Optic'].dropna(subset=['Churn Reason'])).mark_arc().encode(
    theta='count(Customer ID)',
    color=alt.Color('Churn Reason', scale=alt.Scale(scheme='category20')),
    opacity=alt.condition(hover_selection,alt.value(1),alt.value(0.1)),
    tooltip=['Churn Reason','count(Customer ID)']
).properties(
    width=400,
    height=400,
    title='Fiber Optic Customers segmented by Churn Reason'
).add_selection(hover_selection)

st.altair_chart(a, use_container_width=True) 

st.markdown("**Figure 11:** This chart shows us the reasons the people with Fiber Optic tend to churn. We can see that majority of them churn because of competitor related issues. The company may want to improve their Fiber Optic devices and offers for more customer retention. Another major cause of churn seems to be attitude of the service provider, which the company could work on fixing.")

a = alt.Chart(churn_df).mark_bar().encode(
    x='Contract',
    y=alt.Y('count(Customer ID)'),
    tooltip='count(Customer ID)',
    color='Customer Status'
).properties(
    width=600,
    height=500,
    title='Contract type vs Customer status'
)
a

st.markdown("**Figure 12:** This chart shows the status of customers based on their plan. The hghest churn is from customers who pay month-to-month. The company could benefit from incentivizing One/Two year plans.")
#
st.markdown("Let's dive deeper into why Monthly customers churn.")

hover_selection = alt.selection_single(
    clear='mouseout',
    fields=['Churn Reason'],
    name='hover_selection',
    on='mouseover',
)

a = alt.Chart(churn_df[churn_df['Contract'] == 'Month-to-Month'].dropna(subset=['Churn Reason'])).mark_arc().encode(
    theta='count(Customer ID)',
    color=alt.Color('Churn Reason', scale=alt.Scale(scheme='category20')),
    opacity=alt.condition(hover_selection,alt.value(1),alt.value(0.1)),
    tooltip=['Churn Reason','count(Customer ID)']
).properties(
    width=400,
    height=400,
    title='Monthly subscribed customers segmented by Churn Reason'
).add_selection(hover_selection)

st.altair_chart(a, use_container_width=True) 
st.markdown('''**Figure 13:** From the pie chart above, we can clearly see that the monthy paying customer churn mainly due to better plans of competitiors. 
            The attitude of support people are also a major cause for churn.
            Our company should analyze competitors plans and identify why they are more successful. They should also focus on training the support staff.''')

selection = alt.selection_single(fields=['Churn Reason'], bind='legend')

a = alt.Chart(churn_df.dropna(subset=['Churn Category'])).mark_bar().encode(
    x='Churn Category',
    y='count(Customer ID)',
    color=alt.Color('Churn Reason', scale=alt.Scale(scheme='category20')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
    tooltip=['Churn Reason','count(Customer ID)']
).properties(
    width=400,
    height=450,
    title='Overall Churn Categories by Reason'
).add_selection(selection)

st.altair_chart(a, use_container_width=True) 
st.markdown("**Figure 14:** This chart shows the reasons customers churn and puts them into broad categories. Competitiors and bad attitude seem to be the leading causes for people to churn.")

a = alt.Chart(churn_df.groupby('Customer Status')[['Monthly Charge']].mean().reset_index()).mark_bar().encode(
    x='Customer Status',
    y='Monthly Charge',
    tooltip=alt.Tooltip('Monthly Charge',format='$.2f')
).properties(
    width=550,
    height=400,
    title='Average Monthly payment by Customer Status'
)

a

st.markdown('''**Figure 15:** The above bar graph shows the average monthly payment for Churned customers vs Other customers. Clearly, people who churn are paying more than the other customers. From our exploratory analysis, we noticed that out of 1500+ customers churning, only 78 are churning because price is too high.
This is a clear indication that most people who are churning are actually willing to pay for our product, and in fact, pay more than loyal customers on an average.\n
Hence, our model will focus on predicting churn and focus more on recall score since the average value of a churned customer is higher than the rest.''')

total = alt.Chart(churn_df).mark_line(point=True).encode(
    x='Tenure in Months:O',
    y=alt.Y('sum(Monthly Charge)'),
    tooltip=alt.Tooltip('sum(Monthly Charge)',format='$.2f'),
).properties(
    width=750,
    height=350,
    title='Revenue vs months'
)

avg = alt.Chart(churn_df).mark_line(point=True).encode(
    x='Tenure in Months:O',
    y=alt.Y('mean(Monthly Charge)',scale=alt.Scale(domain=[10,90])),
    color=alt.value('green'),
    tooltip=alt.Tooltip('mean(Monthly Charge)',format='$.2f'),
).properties(
    width=750,
    height=350,
    title='Revenue vs months'
)

st.altair_chart((total + avg).resolve_scale(y='independent').configure_point(size=60))

st.markdown('''**Figure 16:** The chart above shows us monthly revenue values. 

The blue line depicts total monthly revenue which drops significantly from Month 1 right up to Month 72. There is a significant drop in the beginning due to high churn, and we can see there is a spike at the end (month 72). The data at month 72 could signify all the current loyal customers. 

The green line charts average monthly revenue per customer. We can see that in spite of losing a lot of customers, the average revenue for our company stays pretty consistent and actually has a slight upward trend.

This indicates that if the company focuses on retaining customers, they could significantly boost their average monthly revenue.''')


def generate_map(temp_df,color):      
      f = folium.Figure(width=1000, height=500)
      map = folium.Map(location=[36.7783,-119.4179], zoom_start=5).add_to(f)
      for idx,row in temp_df.iterrows():
            folium.Circle(
                 location=[row['Latitude'],row['Longitude']],
                 color=color,
                 fill=True,
                 fill_opacity=1,
                 fill_color='red',
                 ).add_to(map)
      return map 


st.subheader("Analyzing Locations of Customers")

# Add radio buttons
churn_status = st.radio(label="",options = ("Churn", "Didn't Churn"))

if churn_status == 'Churn':
      txt = "Churn Locations"
      temp_df = churn_df[churn_df['Customer Status'] == 'Churned']
      color = 'red'
      st.markdown(txt)
      folium_static(generate_map(temp_df, color))
else:
      txt = "No Churn Locations"
      temp_df = churn_df[churn_df['Customer Status'] != 'Churned']
      color = 'green'
      st.markdown(txt)
      folium_static(generate_map(temp_df, color))       
        

st.markdown('''From a high level geospatial analysis, we identify that location does not have much correlation with the churn. Regions with high population density have more churn and more loyal customers and regions with sparse market penetration has lesser churn and lesser stayed customers, simply because the number of customers are low.
\nTherefore, we believe geographic information is not very relevant in predicting churn, we do not use these features to build our model.''')

st.subheader('Correlation Analysis')

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(churn_df.corr(numeric_only=True), annot=True,fmt='.1f',linewidth=.5,cmap=sns.color_palette("rocket_r", as_cmap=True))
st.pyplot(fig,clear_figure=True)

st.markdown('''**Figure 19:** The chart above shows correlations between the numeric features. Some of the more obvious ones are Tenure in Months - with Total Charges, Total Revenue and other charges as well.
\n
An interesting negative correlation is Age with Avg Monthly Downloads in GB. We can see as people grow older, our data reflects that they use lesser internet.''')

st.header("Data Pipeline")

st.subheader("Splitting columns")

st.markdown("Splitting columns into different categories for identification")

st.code('''geo_cols = ['Zip Code', 'Latitude', 'Longitude','City']

numeric_cols = ['Age', 'Number of Dependents', 
       'Number of Referrals', 'Tenure in Months',
       'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
       'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue']

id_col = ['Customer ID']

categorical_cols_yn = ['Device Protection Plan','Gender','Internet Service',
                       'Married','Multiple Lines','Online Backup',
                    'Online Security','Paperless Billing','Phone Service',
                    'Premium Tech Support','Streaming Movies','Streaming Music',
                    'Streaming TV','Unlimited Data']

categorical_cols_3class = ["Contract","Payment Method","Internet Type"]

target_col = ['Customer Status']

other_target_cols = ['Churn Category','Churn Reason','Offer']''')

geo_cols = ['Zip Code', 'Latitude', 'Longitude','City']

numeric_cols = ['Age', 'Number of Dependents', 
       'Number of Referrals', 'Tenure in Months',
       'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download',
       'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue']

id_col = ['Customer ID']

categorical_cols_yn = ['Device Protection Plan','Gender','Internet Service',
                       'Married','Multiple Lines','Online Backup',
                    'Online Security','Paperless Billing','Phone Service',
                    'Premium Tech Support','Streaming Movies','Streaming Music',
                    'Streaming TV','Unlimited Data']

categorical_cols_3class = ["Contract","Payment Method","Internet Type"]

target_col = ['Customer Status']

other_target_cols = ['Churn Category','Churn Reason','Offer']


st.subheader('Creating column transformers')

def replace_yes_no(item):
    return np.where(item == 'Yes', 1, np.where(item == 'No', 0, item))

def encode_gender(item):
    return np.where(item == 'Male', 1, np.where(item == 'Female', 0, item))

def encode_churn(item):
    return np.where(item == 'Churned', 1, np.where((item == 'Stayed') | (item == 'Joined'), 0, item))

st.code('''def replace_yes_no(item):
    return np.where(item == 'Yes', 1, np.where(item == 'No', 0, item))

def encode_gender(item):
    return np.where(item == 'Male', 1, np.where(item == 'Female', 0, item))

def encode_churn(item):
    return np.where(item == 'Churned', 1, np.where((item == 'Stayed') | (item == 'Joined'), 0, item))''')


st.subheader("Pre-processing pipelines")

numerical_pipeline = Pipeline(steps=[
    ('impute0_numeric', SimpleImputer(strategy='constant', fill_value=0)),
    ('minMaxScale_numeric', MinMaxScaler())
    ])

# For yes/no classes
categorical_pipeline1 = Pipeline(steps=[
    ('imputeNo_categoric_yn', SimpleImputer(strategy='constant', fill_value='No')),
    ('ohe_cat_yn', FunctionTransformer(replace_yes_no)),
    ('ohe_cat_gender',FunctionTransformer(encode_gender))
])

# For other classes
categorical_pipeline2 = Pipeline(steps=[
    ('imputeNoInternet_categoric', SimpleImputer(strategy='constant', fill_value='No Internet')),
    ('ohe_cat_3class', OneHotEncoder())
])

# For 'Churn Category','Churn Reason' and 'Offer', for doing multi-label classification
other_target_pipeline = Pipeline(steps=[
    ('impute_target_cols', SimpleImputer(strategy='constant', fill_value='No Churn')),
    ('label_encoder', LabelEncoder())
])

st.code('''numerical_pipeline = Pipeline(steps=[
    ('impute0_numeric', SimpleImputer(strategy='constant', fill_value=0)),
    ('minMaxScale_numeric', MinMaxScaler())
    ])

# For yes/no classes
categorical_pipeline1 = Pipeline(steps=[
    ('imputeNo_categoric_yn', SimpleImputer(strategy='constant', fill_value='No')),
    ('ohe_cat_yn', FunctionTransformer(replace_yes_no)),
    ('ohe_cat_gender',FunctionTransformer(encode_gender))
])

# For other classes
categorical_pipeline2 = Pipeline(steps=[
    ('imputeNoInternet_categoric', SimpleImputer(strategy='constant', fill_value='No Internet')),
    ('ohe_cat_3class', OneHotEncoder())
])

# For 'Churn Category','Churn Reason' and 'Offer', for doing multi-label classification
other_target_pipeline = Pipeline(steps=[
    ('impute_target_cols', SimpleImputer(strategy='constant', fill_value='No Churn')),
    ('label_encoder', LabelEncoder())
])''')

X = churn_df[numeric_cols + categorical_cols_yn + categorical_cols_3class].copy()

y = churn_df[target_col].copy()

y2 = churn_df[other_target_cols].copy()

y = np.array(churn_df['Customer Status'].apply(encode_churn)).astype(int)

st.subheader("Checking class imbalance")

unique, counts = np.unique(y, return_counts=True)
temp = {int(u):int(c) for u,c in zip(unique,counts)}
st.write("Class distribution:", temp)

st.markdown("The target variable is imbalanced, so we will randomly oversample it in the pipeline to resolve this.")

st.subheader("Create Preprocessing Pipeline")

preprocessor = ColumnTransformer(transformers= 
                [('numerical_pipeline', numerical_pipeline, numeric_cols),
                 ('categorical_pipeline1', categorical_pipeline1, categorical_cols_yn),
                 ('categorical_pipeline2', categorical_pipeline2, categorical_cols_3class)
                 ],remainder = 'passthrough')

st.code('''preprocessor = ColumnTransformer(transformers= 
                [('numerical_pipeline', numerical_pipeline, numeric_cols),
                 ('categorical_pipeline1', categorical_pipeline1, categorical_cols_yn),
                 ('categorical_pipeline2', categorical_pipeline2, categorical_cols_3class)
                 ],remainder = 'passthrough')''')

churn_transformed = preprocessor.fit_transform(X)
# churn_transformed[0]

st.header("Building Models")

seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

# print("X train:", X_train.shape)
# print("y train:", y_train.shape)
# print("X test:", X_test.shape)
# print("y test:", y_test.shape)

st.subheader("Binary Classification")
st.markdown("For classifying churn or not churn")

st.caption("Models to try")
clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=seed,max_iter=200)
clf3 = KNeighborsClassifier()
clf4 = svm.SVC()
clf5 = RandomForestClassifier(random_state=seed)
clf6 = xgb.XGBClassifier(random_state=seed,eval_metric="auc")

# Initiaze the hyperparameters for each dictionary
param1 = {}
param1['classifier__var_smoothing'] = [1e-09, 1e-08, 1e-07]
param1['classifier'] = [clf1]

param2 = {}
param2['classifier__C'] = [0.1, 1, 10]
param2['classifier__penalty'] = [None, 'l2']
param2['classifier'] = [clf2]

param3 = {}
param3['classifier__n_neighbors'] = [3, 5, 7]
param3['classifier__weights'] = ['uniform', 'distance']
param3['classifier'] = [clf3]

param4={}
param4['classifier'] = [clf4]
param4['classifier__C'] = [0.1, 1, 10]
param4['classifier__kernel'] = ['rbf','sigmoid']
param4['classifier__gamma'] = [1, 0.1, 0.0001]

param5 = {}
param5['classifier__max_depth'] = [None, 5, 10]
param5['classifier__n_estimators'] = [50, 100, 200]
param5['classifier'] = [clf5]

param6= {}
param6['classifier'] = [clf6]
param6['classifier__eta'] = [0.05,0.25,0.45]
param6['classifier__min_child_weight'] = [10,30,50]


st.code('''clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=seed,max_iter=200)
clf3 = KNeighborsClassifier()
clf4 = svm.SVC()
clf5 = RandomForestClassifier(random_state=seed)
clf6 = xgb.XGBClassifier(random_state=seed,eval_metric="auc")

# Initiaze the hyperparameters for each dictionary
param1 = {}
param1['classifier__var_smoothing'] = [1e-09, 1e-08, 1e-07]
param1['classifier'] = [clf1]

param2 = {}
param2['classifier__C'] = [0.1, 1, 10]
param2['classifier__penalty'] = [None, 'l2']
param2['classifier'] = [clf2]

param3 = {}
param3['classifier__n_neighbors'] = [3, 5, 7]
param3['classifier__weights'] = ['uniform', 'distance']
param3['classifier'] = [clf3]

param4={}
param4['classifier'] = [clf4]
param4['classifier__C'] = [0.1, 1, 10]
param4['classifier__kernel'] = ['rbf','sigmoid']
param4['classifier__gamma'] = [1, 0.1, 0.0001]

param5 = {}
param5['classifier__max_depth'] = [None, 5, 10]
param5['classifier__n_estimators'] = [50, 100, 200]
param5['classifier'] = [clf5]

param6= {}
param6['classifier'] = [clf6]
param6['classifier__eta'] = [0.05,0.25,0.45]
param6['classifier__min_child_weight'] = [10,30,50]''')


st.subheader("Create the model training pipeline")
pipeline = ImbPipeline(steps=[
    ('oversampler', RandomOverSampler()),
    ('preprocessor', preprocessor),
    ('classifier', clf1)
])

params = [param1,param2,param3,param4,param5,param6]

st.code('''pipeline = ImbPipeline(steps=[
    ('oversampler', RandomOverSampler()),
    ('preprocessor', preprocessor),
    ('classifier', clf1)
])

params = [param1, param2, param3, param4, param5, param6]''')

# #This cell takes approximately 5-10 minutes to run

st.subheader('Perform GridSearch over all the models and their hyperparameters')

st.code('''grid_search = GridSearchCV(pipeline, params, cv = 5, n_jobs=-1, scoring='roc_auc',refit=True)
grid_search.fit(X_train, y_train)''')

# st.caption('This may take 5-10 minutes to load')

@st.cache_resource()
def perform_grid_search():
    grid_search = GridSearchCV(pipeline, params, cv = 5, n_jobs=-1, scoring='roc_auc',refit=True)
    grid_search.fit(X_train, y_train)
    return grid_search

grid_search = perform_grid_search()

st.write("Best hyperparameters:", grid_search.best_params_)
st.write("Best Validation score:", grid_search.best_score_)

best_model = grid_search.best_estimator_.named_steps['classifier']

X_test_transformed = preprocessor.transform(X_test)

y_pred = best_model.predict(X_test_transformed)

st.subheader("Evaluating on Test Set")

st.write('Accuracy on test data:',accuracy_score(y_test,y_pred))
st.write('Recall on test data:',recall_score(y_test,y_pred))
st.write('F1 score',f1_score(y_test,y_pred))
st.write()

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=best_model.classes_)
fig, ax = plt.subplots()
disp.plot(ax=ax)
plt.title('Confusion Matrix on Test Data')
st.pyplot(fig,clear_figure=True)

st.markdown('''**Figure 20**: From the confusion matrix, we can see that the model categorized ~ 85.22% (~400) of churners in the test data set.
            This is a good result considering we want to reduce the number of False Negatives (people who churn but we think they won't).
             In those cases, we will not be able to intervene. But, as we can see, the model does a good job of identifying churners on the test data.''')

st.subheader("Best Performing Model Interpretation")

#Get the OHE columns for explainibility
ohe_cols = preprocessor.named_transformers_['categorical_pipeline2'].named_steps['ohe_cat_3class'].get_feature_names_out(categorical_cols_3class)

explainer_cols = numeric_cols + categorical_cols_yn + list(ohe_cols)

explain_df = pd.DataFrame(data = X_test_transformed, columns = explainer_cols)
explain_df = explain_df.apply(pd.to_numeric)


try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(explain_df)
    fig, ax = plt.subplots(figsize=(10,5))
    shap.summary_plot(shap_values,explain_df)
    st.pyplot(plt.gcf(),clear_figure=True)

    st.markdown('''**Figure 21**: The summary plot above allows us to identify the features which have high impact on the model prediction. The color bar on the right depicts the
            actual value of the feature for that observation (row) and the x-axis depicts the impact it has on the output. In this case, a negative impact means this observation
            contributes to the 0 (No Churn) Class.            
            \nFor example, we see that there a significant number of red dots in Number of referrals that have a negative impact on the model output. As we identified before, 
            higher the number of referrals a customer makes, the less likely they are to churn. We see that this is something the model has incorporated.
            \nWe also see a clear distinction in the Month-to-Month contract type where the people with a Month-to-Month contract contribute significantly to the 1 (Churn) Class and the people without, contribute
            significantly to the 0 (No Churn) Class.            
            ''')
except:
     st.markdown("SHAP cannot explain the best performing model.")

st.header("Identifying Probability of churn and determining lost value")
st.markdown("Now that we identified the churners, we want to estimate the business value of retaining them, i.e. how much are they worth? Is it financially wise to spend money to retain them? If yes, how much?")

y_pred_churn_proba = best_model.predict_proba(X_test_transformed)[:,1]

value_df = X_test[['Monthly Charge','Tenure in Months']].copy()

value_df['Churn Probability'] = y_pred_churn_proba

value_df = value_df[y_pred == 1]

# value_df['Churn Risk'] = value_df['Churn Probability'].apply(lambda x : 'Low' if 0 <= x <= 0.33 else 'Medium' if 0.33 < x <= 0.66 else 'High')

value_df['Churn Risk'] = value_df['Churn Probability'].apply(lambda x : 'Medium' if 0.5 <= x < 0.75 else 'High')

st.subheader("Calculating lost revenue of different customer segments based on churn probability")

value_df['Lost Revenue'] = (value_df['Tenure in Months'].max() - value_df['Tenure in Months']) * value_df['Monthly Charge']

st.markdown('''\nLost Revenue is calculated by:\n

Lost Revenue = (72 - Tenure in Months) * Monthly Charge

\n*We choose 72 since 72 months (6 years) is the maximum timeframe provided in the data that we have. Therefore, all our cacluations are over a
6 year timeframe.*''') 

a = alt.Chart(value_df.groupby('Churn Risk').mean()[['Lost Revenue']].reset_index()).mark_bar().encode(
    x='Churn Risk',
    y='Lost Revenue:Q',
    tooltip=alt.Tooltip('Lost Revenue',format="$.2f")
).properties(
    width=500,
    height=500,
    title="Average Lost Revenue by Churn Risk"
)

st.altair_chart(a) 

st.markdown('''**Figure 22:** The bar graph shows us the average lost income by Churn risk. 
            People with higher tendency to churn cause a significantly higher loss in revenue.
            In other words, higher payments seem to be positively correlated to probability of churning.''')


st.subheader("Calculating Present Value of Future Cash Flows")

st.code('''# To calculate PV, we assume a default discount rate of 10% per annum
# You can also adjust it below
discount_rate = 0.1''')

discount_rate = st.slider("Choose discount rate per annum",min_value = 1.0, max_value = 100.0, step = 0.5,value=10.0)

discount_rate = discount_rate / 100

value_df['PV'] = npf.pv((discount_rate / 12),value_df['Tenure in Months'].max() - value_df['Tenure in Months'],-value_df['Monthly Charge'])

st.markdown('''To put it simply, we have taken into account the *time value of money*. 
\nFuture cash flows have a certain value today, called the Present Value which takes into account inflation and other risk factors. 
The discount rate is what incorporates these factors. 
To know more, click [here](https://www.investopedia.com/terms/p/presentvalue.asp#:~:text=Present%20value%20is%20the%20concept,%241%2C000%20five%20years%20from%20now).
\nFor our problem, the current chosen discount rate is {}% per year.'''.format(discount_rate*100))

a = alt.Chart(value_df.groupby('Churn Risk').mean()[['PV']].reset_index()).mark_bar().encode(
    x='Churn Risk',
    y='PV:Q',
    tooltip=alt.Tooltip('PV',format="$.2f")
).properties(
    width=400,
    height=450,
    title="Average Present Value of Future Cash Flows by Churn Risk"
)

st.altair_chart(a)

st.markdown('''**Figure 23:** The bar graph shows us the Present Value of Future Cash Flows categorized by Churn Risk.

Present value is calculated using the pv() function in the numpy-financial library

In summary, the PV of a customer or the average PV of a group tells us the value of that individual or a 
group staying with the company, assuming the revenue we generate from them *remains the same*. 
Therefore, to retain that person or group, the PV should be the maximum amount the company is willing to spend. 
Spending any more would result in losses in the long term.''')

grouped_val = value_df.groupby(['Tenure in Months','Churn Risk']).agg({'PV': 'mean','Lost Revenue':'count'})
grouped_val.reset_index(inplace=True)
grouped_val.columns= ['Tenure in Months','Churn Risk','Mean PV','Count']
# grouped_val['Total'] = grouped_val.groupby('Tenure in Months')['Count'].transform('sum')
# grouped_val['Percent'] = grouped_val['Count'] / grouped_val['Total']

# grouped_val.head()

a = alt.Chart(grouped_val).mark_bar().encode(
    x=alt.X('Tenure in Months',bin=alt.Bin(maxbins=100)),
    y='Mean PV:Q',
    color='Churn Risk',
    tooltip=alt.Tooltip('Mean PV',format="$.2f")
).properties(
    width=800,
    height=350,
    title="Average Present Value of Future Cash Flows categorized into Churn Risk by Tenure in Months"
)

st.altair_chart(a)

st.markdown('''**Figure 24:** The bar chart above plots the mean PV of a customer based on their Churn Risk by their tenure in months. 
            We can clearly see that PV's are higher in the initial months and reduce as we move towards the 72 month mark. 
            This graph tells us at each point, the maximum amount we should spend to retain a customer by their Churn Risk 
            assuming a discount rate of {}% per year and that the revenue we will generate from the customer, stays the same as it has 
            been historically up until that point.'''.format(discount_rate*100))

st.header("Multi-label Classification")
st.subheader("Predicting churn reason for identified churners")
st.markdown('''Now we aim to filter out those who churned and to analyze and predict the underlying reasons for them churning
Since we already created train-test splits, we will be using the same ones to train our new multi-label classifier model.
If we redo the split, we will have data leakage issues, since some of the customers in our first set test 
(for whom we are trying to predict churn/no churn), may be in the train split for our second model. 
So, we will just use our previous splits to train our new model.''')

#Merge X_train and X_test with original data based on indexes
churn_X_train = pd.merge(X_train, churn_df, how='inner', left_index=True, right_index=True, suffixes=('', '_right'))
churn_X_test = pd.merge(X_test, churn_df, how='inner', left_index=True, right_index=True, suffixes=('', '_right'))

#Drop duplicate columns based on added suffix '_right'
churn_X_train = churn_X_train.filter(regex='^(?!.*_right)')
churn_X_test = churn_X_test.filter(regex='^(?!.*_right)')

#Keep only rows with churned customers, here we use our prediction
churn_X_train = churn_X_train[y_train==1]
churn_X_test = churn_X_test[y_pred==1]

#Get the y columns, in this case churn category we are trying to predict
churn_y_train = churn_X_train['Churn Category']
churn_y_test = churn_X_test['Churn Category']

#Keep only the needed columns in the churn_X
churn_X_train = churn_X_train[numeric_cols + categorical_cols_yn + categorical_cols_3class].copy()
churn_X_test = churn_X_test[numeric_cols + categorical_cols_yn + categorical_cols_3class].copy()


st.code('''# Merge X_train and X_test with original data based on indexes to get all rows and features we need
churn_X_train = pd.merge(X_train, churn_df, how='inner', left_index=True, right_index=True, suffixes=('', '_right'))
churn_X_test = pd.merge(X_test, churn_df, how='inner', left_index=True, right_index=True, suffixes=('', '_right'))

# Drop duplicate columns based on added suffix '_right'
churn_X_train = churn_X_train.filter(regex='^(?!.*_right)')
churn_X_test = churn_X_test.filter(regex='^(?!.*_right)')

# Keep only rows with churned customers, here we use our prediction
churn_X_train = churn_X_train[y_train==1]
churn_X_test = churn_X_test[y_pred==1]

# Get the y columns, in this case churn category we are trying to predict
churn_y_train = churn_X_train['Churn Category']
churn_y_test = churn_X_test['Churn Category']

# Keep only the needed columns in the churn_X
churn_X_train = churn_X_train[numeric_cols + categorical_cols_yn + categorical_cols_3class].copy()
churn_X_test = churn_X_test[numeric_cols + categorical_cols_yn + categorical_cols_3class].copy()''')


# # print(churn_X_train.shape)
# # print(churn_y_train.shape)
# # print(churn_X_test.shape)
# # print(churn_y_test.shape)

st.markdown("The new dataset looks like:")

st.write(churn_X_train.head().T)

st.caption("We maintain the indices from the original data to clearly distinguish the customer since we drop the ID for modelling purposes.")

st.subheader("Checking for Imbalance")

unique, counts = np.unique(churn_y_train, return_counts=True)
st.write("Class distribution:", {u:int(c) for u,c in zip(unique, counts)})

resample = RandomOverSampler()
churn_X_train_resampled, churn_y_train_resampled = resample.fit_resample(churn_X_train, churn_y_train)

unique, counts = np.unique(churn_y_train_resampled, return_counts=True)

le = LabelEncoder()
churn_y_train_labelencoded = le.fit_transform(churn_y_train_resampled)

churn_y_train_resampled.unique()

st.subheader("Building Models")

st.caption('''Models to try for multi-label classification
           These are similar models and hyperparameters that we used above for binary classification''')
clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=seed,max_iter=200)
clf3 = KNeighborsClassifier()
clf4 = svm.SVC()
clf5 = RandomForestClassifier(random_state=seed)
clf6 = xgb.XGBClassifier(random_state=seed,eval_metric="error")

st.code('''clf1 = GaussianNB()
clf2 = LogisticRegression(random_state=seed,max_iter=200)
clf3 = KNeighborsClassifier()
clf4 = svm.SVC()
clf5 = RandomForestClassifier(random_state=seed)
clf6 = xgb.XGBClassifier(random_state=seed,eval_metric="error")''')

# Initiaze the hyperparameters for each dictionary
param1 = {}
param1['classifier__var_smoothing'] = [1e-09, 1e-08, 1e-07]
param1['classifier'] = [clf1]

param2 = {}
param2['classifier__C'] = [0.1, 1, 10]
param2['classifier__penalty'] = [None, 'l2']
param2['classifier'] = [clf2]

param3 = {}
param3['classifier__n_neighbors'] = [3, 5, 7]
param3['classifier__weights'] = ['uniform', 'distance']
param3['classifier'] = [clf3]

param4={}
param4['classifier'] = [clf4]
param4['classifier__C'] = [0.1, 1, 10]
param4['classifier__kernel'] = ['rbf','sigmoid']
param4['classifier__gamma'] = [1, 0.1, 0.0001]

param5 = {}
param5['classifier__max_depth'] = [None, 5, 10]
param5['classifier__n_estimators'] = [50, 100, 200]
param5['classifier'] = [clf5]

param6= {}
param6['classifier'] = [clf6]
param6['classifier__eta'] = [0.05,0.25,0.45]
param6['classifier__min_child_weight'] = [10,30,50]


pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf1)
])

params = [param1, param2, param3, param4, param5, param6]


st.code('''grid_search2 = GridSearchCV(pipeline, params, cv = 5, n_jobs=-1, scoring='accuracy',refit=True)
grid_search2.fit(churn_X_train_resampled, churn_y_train_labelencoded)''')
st.caption("This grid search also uses similar models and hyperparameters. However, we use 'accuracy' to identify the best model")

@st.cache_resource()
def perform_multilabel_grid_search():
    grid_search2 = GridSearchCV(pipeline, params, cv = 5, n_jobs=-1, scoring='accuracy',refit=True)
    grid_search2.fit(churn_X_train_resampled, churn_y_train_labelencoded)
    return grid_search2

grid_search2 = perform_multilabel_grid_search()

# Print the best hyperparameters and test accuracy
st.write("Best hyperparameters:", grid_search2.best_params_)
st.write("Best Validation accuracy:", grid_search2.best_score_)

churn_best_model = grid_search2.best_estimator_.named_steps['classifier']

churn_X_test_transformed = preprocessor.transform(churn_X_test)

churn_y_pred = churn_best_model.predict(churn_X_test_transformed)
churn_y_pred_proba = churn_best_model.predict_proba(churn_X_test_transformed)

st.code('''
# We predict the probabilities of every churn category for a given churned customer
churn_y_pred_proba = churn_best_model.predict_proba(churn_X_test_transformed)''')

# getting the probability of the churn reasons across five different churning categories for every churned customers

# print(churn_y_pred[0:5])
# print(churn_y_pred_proba[0:5])

# adding the probability for the most and the second likely category to each observation
churn_y_pred_max_proba = [i.max() for i in churn_y_pred_proba]
churn_y_pred_sec_proba = [i[np.argsort(i)[-2]] for i in churn_y_pred_proba]
churn_y_pred_sec_category = [np.argsort(i)[-2] if i[np.argsort(i)[-1]] != i[np.argsort(i)[-2]] else np.argsort(i)[-1] for i in churn_y_pred_proba]
# print(churn_y_pred_max_proba[0:5])
# print(churn_y_pred_sec_proba[0:5])
# print(churn_y_pred_sec_category[0:5])

churn_X_test['Prob_1'] = churn_y_pred_max_proba
churn_X_test['Category_1'] = churn_y_pred
churn_X_test['Prob_2'] = churn_y_pred_sec_proba
churn_X_test['Category_2'] = churn_y_pred_sec_category

# print(churn_X_test.shape)
churn_X_test = churn_X_test.reset_index(drop=True)
# churn_X_test.head().T

d = {0:'Attitude', 1:'Competitor', 2:'Dissatisfaction', 3:'Other', 4:'Price'}

churn_X_test['Churn Category_1'] = churn_X_test['Category_1'].map(d)
churn_X_test['Churn Category_2'] = churn_X_test['Category_2'].map(d)

chart = alt.Chart(churn_X_test).mark_bar().encode(
    x='Churn Category_1:N',
    y='count()',
    color = 'Churn Category_1:N',
    tooltip = ['Churn Category_1','count()']
).properties(
    width=800,
    height=550,
    title="Number of the Most Likely Churning Category for Churned Customers"
)

st.altair_chart(chart)

st.markdown('''**Figure 25**: The chart above shows us the number of churners by their primary reason of predicted churn.
Clearly, we see that for most customers, the model suggests that churning to competitors has the highest probability.''')


a = alt.Chart(churn_X_test).mark_bar().encode(
    x='Churn Category_1:N',
    y='sum(Total Revenue)',
    color='Churn Category_1:N',
    tooltip=alt.Tooltip('sum(Total Revenue)',format="$.2f")
     ).properties(
    width=800,
    height=550,
    title="Lost Revenue of Each Churning Category for Churned Customers"
)
st.altair_chart(a)


st.markdown('''**Figure 26**: The chart above displays the lost revenue for every primary reason of predicted churn.
The amount of revenue the company loses to its customers is a whopping $1,016,992''')


chart = alt.Chart(churn_X_test).mark_bar().encode(
    x='Churn Category_2:N',
    y='count()',
    color = 'Churn Category_2:N',
    tooltip = ['Churn Category_2','count()']
).properties(
    width=800,
    height=550,
    title="Number of the Second Most Likely Churning Category for Churned Customers"
)

st.altair_chart(chart)

st.markdown('''**Figure 27**: The chart above shows us the number of churners by their secondary reason of predicted churn.
This chart is a little less skewed, but the second highest reason for churn for most customers is dissatisfaction with the company, which is dissatisfaction with the 
service, product, network issues etc.''')

# a = alt.Chart(churn_X_test).mark_bar().encode(
#     x = alt.X('Churn Category_1:N'),
#     y = alt.Y('count()'),
#     color = 'Contract:N',
#     tooltip=alt.Tooltip('count()')
# ).properties(
#     width=550,
#     height=450,
#     title = 'Number of Different Contracts in each Churning Category'
# )

# a

a = alt.Chart(churn_X_test).mark_bar().encode(
    x = alt.X('Churn Category_1:N'),
    y = alt.Y('count()'),
    color = 'Payment Method:N',
    tooltip=alt.Tooltip('count()')
).properties(
    width=500,
    height=450,
    title='Payment Method to each Customers\' Most Likely Churning Category'
)

b = alt.Chart(churn_X_test).mark_bar().encode(
    x = alt.X('Churn Category_2:N'),
    y = alt.Y('count()'),
    color = 'Payment Method:N',
    tooltip=alt.Tooltip('count()')
).properties(
    width=500,
    height=450,
    title='Payment Method to each Customers\' Second Most Likely Churning Category'
)

c = alt.vconcat(a,b)

st.altair_chart(c)

st.markdown('''**Figure 28**: This chart shows us the payment methods of churners for the highest and second highest predicted probability of churn.
An interesting observation here is that most people pay with 'Bank Withdrawls'. As of today, when everything is digital, the company should look into making it easier for the
customer to make their payments.''')


st.header("Conclusion and Key Takeaways")
st.markdown('''
* Based on the data and our predictive model, we can identify churners with a recall of 0.85.
* Identifying the long term value of the customers based on their previous payments, 
we can provide insights and upper bounds on the marketing budgets for customers who are predicted to be churners,
based on segmentation by churn probability.
* Customers with a higher probability of churn tend to pay more. Offering them discounts on their plans could potentially motivate them to stay.
* Predicting churn category with an accuracy of 0.89, the most probable reason of churn is 'Competitor' and the second highest is 'Dissatisfaction'. The company
should analyze what their competitors are offering that they cannot, and improve on the shortfalls in their existing services and products.
This would address the reasons for majority of customers who churn.
''')

st.subheader("References")
st.markdown('''Data Source: https://data.world/mcc450/telecom-churn
\n Churn Statistics:
\n* https://hbr.org/2022/12/in-a-downturn-focus-on-existing-customers-not-potential-ones
\n* https://www.techmahindra.com/en-in/blog/maximizing-business-profit-through-telecom-analytics-solutions/
\n* https://www.statista.com/statistics/816735/customer-churn-rate-by-industry-us/
\nPipeline Code references:
\n* http://glemaitre.github.io/imbalanced-learn/generated/imblearn.pipeline.Pipeline.html
\n* https://scikit-learn.org/stable/modules/compose.html
\nGeneral Numpy code: https://numpy.org/doc/
\nNumpy Financial Code for NPV: https://numpy.org/numpy-financial/latest/
\nStreamlit Code: https://docs.streamlit.io/
\nAltair Visualizations: https://community.altair.com/community?id=altair_product_documentation
''')

# BELOW CODE IS NOT SHOWN ON APP
# Find its implementation in the notebook

# a = alt.Chart(churn_X_test).mark_bar().encode(
#     x = alt.X('Churn Category_1:N'),
#     y = alt.Y('count()'),
#     color = 'Internet Type:N',
#     tooltip=alt.Tooltip('count()')
# ).properties(
#     width=550,
#     height=450,
#     title='Internet Type to each Customers\' Most Likely Churning Category'
# )

# a

# a = alt.Chart(churn_X_test).mark_bar().encode(
#     x = alt.X('Churn Category_1:N'),
#     y = alt.Y('count()'),
#     color = 'Unlimited Data:N',
#     tooltip=alt.Tooltip('count()')
# ).properties(
#     width=550,
#     height=450,
#     title='Unlimited Data to each Customers\' Most Likely Churning Category'
# )

# a


# a = alt.Chart(churn_X_test).mark_boxplot().encode(
#     x = 'Churn Category_1:N',
#     y = 'Prob_1:Q',
#     color = 'Churn Category_1:N'
# ).properties(
#     title='Distribution of Probability of each Customers\' Most Likely Churning Category',
#     width=600,
#     height=500
# )

# st.altair_chart(a)

# a = alt.Chart(churn_X_test).mark_boxplot().encode(
#     x = 'Churn Category_2:N',
#     y = 'Prob_2:Q',
#     color = 'Churn Category_2:N'
# ).properties(
#     title='Distribution of Probability of each Customers\' Second Most Likely Churning Category',
#     width=600,
#     height=500
# )

# st.altair_chart(a)



# line = alt.Chart(churn_X_test).mark_line().encode(
#     x='Prob_1:Q',
#     y='count()',
#     color='Churn Category_1:N',
# ).properties(
#     width=800,
#     height=350,
#     title="Probability of the Most Likely Churning Category for Churned Customers"
# )

# points = alt.Chart(churn_X_test).mark_circle().encode(
#     x='Prob_1:Q',
#     y='count()',
#     color='Churn Category_1:N',
#     tooltip=alt.Tooltip('Prob_1')
# )

# st.altair_chart((line + points))

# ## Calculate Lost Revenue for each Churning Category

#{0:'Attitude', 1:'Competitor', 2:'Dissatisfaction', 3:'Other', 4:'Price'}
# churn_value_df = churn_X_test[['Monthly Charge','Tenure in Months','Prob_1','Category_1','Prob_2','Category_2']].copy()

# churn_value_attitude_df = churn_value_df[churn_value_df['Category_1'] == 0].copy()
# churn_value_competitor_df = churn_value_df[churn_value_df['Category_1'] == 1].copy()
# churn_value_dissatisfaction_df = churn_value_df[churn_value_df['Category_1'] == 2].copy()
# churn_value_price_df = churn_value_df[churn_value_df['Category_1'] == 4].copy()

# churn_value_attitude_df['Churn Risk'] = churn_value_attitude_df['Prob_1'].apply(lambda x : 'Medium' if 0.2 <= x < 0.4 else 'High')
# churn_value_competitor_df['Churn Risk'] = churn_value_competitor_df['Prob_1'].apply(lambda x : 'Medium' if 0.2 <= x < 0.4 else 'High')
# churn_value_dissatisfaction_df['Churn Risk'] = churn_value_dissatisfaction_df['Prob_1'].apply(lambda x : 'Medium' if 0.2 <= x < 0.4 else 'High')
# churn_value_price_df['Churn Risk'] = churn_value_price_df['Prob_1'].apply(lambda x : 'Medium' if 0.2 <= x < 0.4 else 'High')

# churn_value_competitor_df.head()

# churn_value_competitor_df['Lost Revenue'] = (value_df['Tenure in Months'].max() - churn_value_competitor_df['Tenure in Months']) * churn_value_competitor_df['Monthly Charge']

# a = alt.Chart(churn_value_competitor_df.groupby('Churn Risk').mean()[['Lost Revenue']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='Lost Revenue:Q',
#     tooltip=alt.Tooltip('Lost Revenue',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Lost Revenue by Churn Risk of Category 'Competitor'"
# )

# a

# churn_value_competitor_df['PV'] = npf.pv((discount_rate / 12),value_df['Tenure in Months'].max() - churn_value_competitor_df['Tenure in Months'],-churn_value_competitor_df['Monthly Charge'])

# a = alt.Chart(churn_value_competitor_df.groupby('Churn Risk').mean()[['PV']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='PV:Q',
#     tooltip=alt.Tooltip('PV',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Present Value of Future Cash Flows by Churn Risk of Category \'Competitor'"
# )

# a

# churn_value_attitude_df['Lost Revenue'] = (value_df['Tenure in Months'].max() - churn_value_attitude_df['Tenure in Months']) * churn_value_attitude_df['Monthly Charge']

# a = alt.Chart(churn_value_attitude_df.groupby('Churn Risk').mean()[['Lost Revenue']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='Lost Revenue:Q',
#     tooltip=alt.Tooltip('Lost Revenue',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Lost Revenue by Churn Risk of Category 'Attitude'"
# )

# a


# churn_value_attitude_df['PV'] = npf.pv((discount_rate / 12),value_df['Tenure in Months'].max() - churn_value_attitude_df['Tenure in Months'],-churn_value_attitude_df['Monthly Charge'])

# a = alt.Chart(churn_value_attitude_df.groupby('Churn Risk').mean()[['PV']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='PV:Q',
#     tooltip=alt.Tooltip('PV',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Present Value of Future Cash Flows by Churn Risk of Category \'Attitude'"
# )

# a

# churn_value_dissatisfaction_df['Lost Revenue'] = (value_df['Tenure in Months'].max() - churn_value_dissatisfaction_df['Tenure in Months']) * churn_value_dissatisfaction_df['Monthly Charge']

# a = alt.Chart(churn_value_dissatisfaction_df.groupby('Churn Risk').mean()[['Lost Revenue']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='Lost Revenue:Q',
#     tooltip=alt.Tooltip('Lost Revenue',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Lost Revenue by Churn Risk of Category 'Dissatisfaction'"
# )

# a

# churn_value_dissatisfaction_df['PV'] = npf.pv((discount_rate / 12),value_df['Tenure in Months'].max() - churn_value_dissatisfaction_df['Tenure in Months'],-churn_value_dissatisfaction_df['Monthly Charge'])

# a = alt.Chart(churn_value_dissatisfaction_df.groupby('Churn Risk').mean()[['PV']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='PV:Q',
#     tooltip=alt.Tooltip('PV',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Present Value of Future Cash Flows by Churn Risk of Category \'Dissatisfaction'"
# )

# a

# churn_value_price_df['Lost Revenue'] = (value_df['Tenure in Months'].max() - churn_value_price_df['Tenure in Months']) * churn_value_price_df['Monthly Charge']

# a = alt.Chart(churn_value_price_df.groupby('Churn Risk').mean()[['Lost Revenue']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='Lost Revenue:Q',
#     tooltip=alt.Tooltip('Lost Revenue',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Lost Revenue by Churn Risk of Category 'Price'"
# )

# a

# churn_value_price_df['PV'] = npf.pv((discount_rate / 12),value_df['Tenure in Months'].max() - churn_value_price_df['Tenure in Months'],-churn_value_price_df['Monthly Charge'])

# a = alt.Chart(churn_value_price_df.groupby('Churn Risk').mean()[['PV']].reset_index()).mark_bar().encode(
#     x='Churn Risk',
#     y='PV:Q',
#     tooltip=alt.Tooltip('PV',format="$.2f")
# ).properties(
#     width=300,
#     height=300,
#     title="Average Present Value of Future Cash Flows by Churn Risk of Category \'Price'"
# )

# a
