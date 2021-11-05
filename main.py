import streamlit as st
import pandas as pd
import time

st.image('Machine Learner.png')

data = st.file_uploader("Upload your data here")

if data:
    my_dataframe = pd.read_csv(data)
    st.dataframe(my_dataframe)
    features = list(my_dataframe)
    target = st.multiselect('Select target', features)


classifiers = ['Random Forest', 'Decision Tree', 'K Nearest Neighbors']

selected_classifiers = st.multiselect('Select classifiers', classifiers)

if 'Random Forest' in selected_classifiers:
    st.balloons()

for clf in selected_classifiers:
    st.write(clf)

wants_pca = st.radio('U wanna PCA?', ['Yup', 'Nope'])


