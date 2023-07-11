#-----------------------#
# IMPORT DES LIBRAIRIES #
#-----------------------#

import streamlit as st
import joblib
import plotly.graph_objects as go
import matplotlib as plt
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import shap
import requests as re
import numpy as np
import pandas as pd


#---------------------#
# VARIABLES STATIQUES #
#---------------------#

#API_PRED = "https://api-creditscore.herokuapp.com/predict/"
#API_SHAP = "https://api-creditscore.herokuapp.com/shap_client/"
API_PRED = "http://127.0.0.1:8000/predict/"
API_SHAP = "http://127.0.0.1:8000/shap_client/"

data = joblib.load('sample_test_set.pickle')
infos_client = joblib.load('infos_client.pickle')
pret_client = joblib.load('pret_client.pickle')
preprocessed_data = joblib.load('preprocessed_data.pickle')
model = joblib.load('model.pkl')

column_names = preprocessed_data.columns.tolist()
expected_value = -2.9159221699244515
threshold = 100-10.344827586206896

classifier = model.named_steps['classifier']
df_preprocess = model.named_steps['preprocessor'].transform(data)
explainer = shap.TreeExplainer(classifier)
generic_shap = explainer.shap_values(df_preprocess, check_additivity=False)

html="""           
    <h1 style="font-size:400%; color:DarkSlateGrey; font-family:Soleil"> DASHBOARD <br>
        <body style="font-size:100%, color:DarkSlateGrey, font-family:Sofia Pro"> <br>
        </body>
     </h1>
"""
st.markdown(html, unsafe_allow_html=True)

#---------#
# SIDEBAR #
#---------#

#Profile Client
profile_ID = st.sidebar.selectbox('Sélectionnez un client :',
                                  list(data.index))
API_GET = API_PRED+(str(profile_ID))
score_client = 100-int(re.get(API_GET).json()*100)
if score_client < threshold:
    st.sidebar.write("Prêt refusé")
else:
    st.sidebar.write("Prêt accordé.")

# Affichage de la jauge
jauge = go.Figure(go.Indicator(
                  mode='gauge+number+delta',
                  value=score_client,
                  domain={'x': [0, 1], 'y': [0, 1]},
                  gauge={'axis': {'range': [None,100],
                                  'tickwidth': 3,
                                  'tickcolor': 'DarkSlateGrey'},
                         'bar': {'color': '#F0F2F6', 'thickness': 1},
                         'steps': [{'range': [0,50], 'color': 'Crimson'},
                                   {'range': [50,70], 'color': 'Orange'},
                                   {'range': [70,100-10.344827586206896], 'color': 'red'},
                                   {'range': [100-10.344827586206896,95], 'color': 'LimeGreen'},
                                   {'range': [95,100], 'color': 'Green'}],
                         'threshold': {'line': {'color': 'DarkSlateGrey', 'width': 2},
                                       'thickness': 1,
                                       'value' : threshold}}))

jauge.update_layout(height=250, width=305,
                    font={'color': 'black', 'family': 'Sofia Pro'},
                    margin=dict(l=0, r=0, b=0, t=0, pad=2))

st.sidebar.plotly_chart(jauge)
if 95 <= score_client < 100:
    score_text = 'DOSSIER PARFAIT'
    st.sidebar.success(score_text)
elif threshold <= score_client < 95:
    score_text = 'DOSSIER CORRECT'
    st.sidebar.success(score_text)
elif 70 <= score_client < threshold:
    score_text = 'DOSSIER À RÉVISER'
    st.sidebar.warning(score_text)
else :
    score_text = 'DOSSIER INSOLVABLE'
    st.sidebar.error(score_text)

#---------------------------------------#
# INFORMATIONS GÉNÉRIQUES SUR LE CLIENT #
#---------------------------------------#

# Infos principales client
html="""
     <h3 style="font-size:150%; color:DarkSlateGrey; font-family:Sofia Pro"> Récapitulatif du profil<br>
     </h3>
"""
st.markdown(html, unsafe_allow_html=True)
client_info = infos_client[infos_client.index == profile_ID].iloc[:, :]
st.table(client_info)

# Infos principales sur la demande de prêt
html="""
     <h3 style="font-size:150%; color:DarkSlateGrey; font-family:Sofia Pro"> Caractéristiques du prêt<br>
     </h3>
"""
st.markdown(html, unsafe_allow_html=True)
client_pret = pret_client[pret_client.index== profile_ID].iloc[:, :]
st.table(client_pret)


#-----------------------#
# CRÉATION DE VARIABLES #
#-----------------------#

html = """
         <h3 style="font-size:150%; color:DarkSlateGrey; font-family:Sofia Pro"> Contribution des variables au modèle <br>
         </h3>
    """
st.markdown(html, unsafe_allow_html=True)

with st.container():
    col1,col2 = st.columns([4,3.3])
    with col1:
        # Interprétabilité locale SHAP
        st.write("Pour le client {}:".format(profile_ID))
        API_GET = API_SHAP + (str(profile_ID))
        shap_values = re.get(API_GET).json()
        shap_values = np.array(shap_values[1:-1].split(',')).astype('float32')
        waterfall = shap.plots._waterfall.waterfall_legacy(expected_value=expected_value,
                                                           shap_values=shap_values,
                                                           feature_names=column_names,
                                                           max_display=20)
        st.pyplot(waterfall)
    with col2:
        # Interprétabilité globale SHAP
        st.write("Pour l'ensemble des clients:")
        # shap.plots.bar(dict(shap_values), max_display=40)
        summary = shap.summary_plot(shap_values=generic_shap,
                                    feature_names=column_names,
                                    max_display=20)
        st.pyplot(waterfall)

# Graphique interactifs
html = """
             <h3 style="font-size:150%; color:DarkSlateGrey; font-family:Sofia Pro"> Classification graphique interactive <br>
             </h3>
        """
st.markdown(html, unsafe_allow_html=True)
features = st.multiselect("Choisissez deux variables",
                          list(data.columns),
                          default=['AMT_ANNUITY','AMT_INCOME_TOTAL'],
                          max_selections=2)
if len(features) != 2:
    st.error("Sélectionnez deux variables")
else:
    # Graphique
    chart = px.scatter(data,
                       x=features[0],
                       y=features[1],
                       color='TARGET',
                       color_discrete_sequence=['LimeGreen','Tomato'],
                       hover_name=data.index)
    st.plotly_chart(chart)
