import streamlit as st
from pycaret.classification import *
import pandas as pd
import numpy as np


#应用主题
st.set_page_config(
    page_title="ML Medicine",
    #page_icon="🐇",
)
#隐藏选项卡
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


#应用标题
st.title('Machine Learning Application for Predicting Poor Recovery of Functional Range of Motion')

Age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25,)
BMI = st.sidebar.number_input("BMI", min_value=10.00, max_value=50.00, value=22.00,)
Length_of_leg = st.sidebar.number_input("Length of leg (cm)", min_value=1, max_value=200, value=88,)
ROM_0W = st.sidebar.number_input("Preoperative ROM (°)", min_value=1, max_value=200, value=110,)
IKDC = st.sidebar.number_input("IKDC score", min_value=1.00, max_value=100.00, value=52.00,)
LEFS = st.sidebar.number_input("LEFS score", min_value=1.00, max_value=100.00, value=56.00,)
SF12_PCS = st.sidebar.number_input("SF-12 PCS score", min_value=1.00, max_value=100.00, value=37.00,)
SF12_MCS = st.sidebar.number_input("SF-12 PCS score", min_value=1.00, max_value=100.00, value=49.00,)
# Age = st.sidebar.slider("age",0,100,50,1)
# Hypertension = st.sidebar.selectbox("Hypertension",('No','Yes'))
# CHD = st.sidebar.selectbox("CHD",('No','Yes'))
# Lipid_disorder = st.sidebar.selectbox("Lipid_disorder",('No','Yes'))
# Stroke = st.sidebar.selectbox("Stroke",('No','Yes'))
# Heart_failure = st.sidebar.selectbox("Heart_failure",('No','Yes'))
# Cancer = st.sidebar.selectbox("Cancer",('No','Yes'))
# Diabetes = st.sidebar.selectbox("Diabetes",('No','Yes'))
# COPD = st.sidebar.selectbox("COPD",('No','Yes'))
# Chronic_kidney_disease = st.sidebar.selectbox("Chronic_kidney_disease",('No','Yes'))

#映射字典
# map = {'No':0,'Yes':1}

#映射



# male_gender = map[male_gender]
# Hypertension = map[Hypertension]


#读之前存储的模型
model = load_model('saved_xgb')

#建立输入框
input_dict = {'Age':Age, 'BMI':BMI, 'Length_of_leg':Length_of_leg,
               'IKDC':IKDC, 'LEFS':LEFS,
              'SF12_PCS':SF12_PCS, 'SF12_MCS':SF12_MCS,'ROM_0W':ROM_0W,
              }
input_df = pd.DataFrame([input_dict])

###########################
# st.write(input_df)

#画shap##############################
prob = round(model.predict_proba(input_df)[0][1],4)
col1, col2 = st.columns(2)
import shap

# dataset = pd.read_excel("sample.xlsx")
dataset = pd.read_csv('ros_train.csv')

features = ['Age', 'BMI', 'Length_of_leg', 'IKDC',
            'LEFS', 'SF12_PCS','SF12_MCS', 'ROM_0W',]
X = dataset[features]
y = dataset['ROM_12W']
# X = X.loc[0:100]
#X = X.sample(n=100,random_state=123)
#这里要多加1
# y = y[0:101]
#X_summary = shap.kmeans(X,10)

X_expand = pd.concat([X,input_df], ignore_index = True)
y_expand = np.hstack([y,model.predict(input_df)])

from explainerdashboard import ClassifierExplainer

explainer = ClassifierExplainer(model, X,y,)

#画概率值显示
# col2.write(explainer.plot_roc_auc())
#############
col2.write('# ')
col2.metric(label="Risk Probability", value='{:.2%}'.format(prob),)

#画饼
import plotly.express as px
pie_dic={'Poor_functional_ROM':[prob,1-prob],'label':['Yes','No']}
pie_df=pd.DataFrame(pie_dic)
fig = px.pie(pie_df,names='label' ,values='Poor_functional_ROM',color='label',
             color_discrete_map={'Yes':'#db4052','No':'#73a6d2'},
             hole=.35,title='Poor Functional ROM',width=500,height=500,
            )
col1.write(fig)

#截断值
sp = 0.5
is_t = prob > sp
# is_t = (model.predict_proba(input_df)[0][1])> sp

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'


st.markdown('### Risk grouping:  '+str(result))
st.write('# ')
#######
expander = st.expander("See Result Explanation by Individual Features Contribution")

explainer = ClassifierExplainer(model, input_df,X_background=X)
#st.write(X_expand)
#st.write(len(X))
expander.write(explainer.plot_contributions(index=0,#len(X)
                                      sort='high-to-low',#sort ( {'abs' , 'high-to-low' , 'low-to-high' , 'importance'}
                                      orientation='horizontal',#{'vertical', 'horizontal'}
                                      ))

st.sidebar.button('Click or Enter')
#################################
# #截断点
# sp = 0.5
# #figure
# is_t = (model.predict_proba(input_df)[0][1])> sp
# prob = round(model.predict_proba(input_df)[0][1],4)
#
# st.write('{:.2%}'.format(prob))
# #预测
# if is_t:
#     result = 'High Risk'
# else:
#     result = 'Low Risk'
# if st.button('Predict'):
#     st.markdown('## Risk grouping:  '+str(result))
#     if result == 'Low Risk':
#         st.balloons()
#     st.markdown('## Probability:  '+str(round(prob*100,4))+'%')
