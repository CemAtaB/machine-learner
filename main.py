import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

classifiers = {
    'Random Forest': {'model': RandomForestClassifier(random_state=0),
                      'info': open("random-forest-info.txt", "r").read()},
    'Decision Tree': {'model': DecisionTreeClassifier(random_state=0),
                      'info': open("decision-tree-info.txt", "r").read()},
    'K Nearest Neighbor': {'model': KNeighborsClassifier(),
                           'info': open("knn-info.txt", "r").read()}
}

regressors = {
    'Random Forest': {'model': RandomForestRegressor(random_state=0),
                      'info': open("random-forest-info.txt", "r").read()},
    'Decision Tree': {'model': DecisionTreeRegressor(random_state=0),
                      'info': open("decision-tree-info.txt", "r").read()},
    'K Nearest Neighbor': {'model': KNeighborsRegressor(),
                           'info': open("knn-info.txt", "r").read()}
}


@st.cache
def convert_df(data_frame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return data_frame.to_csv().encode('utf-8')


# Harry B. ,Retrieved from : https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python/52935270
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res


def remove_features():
    df.drop(features_to_remove, axis=1, inplace=True)
    data_showing.dataframe(df)


chosen_model = None
df = None
original_df = None
st.image('Machine Learner.png')

data = st.sidebar.file_uploader("Upload your data here")

try:
    original_df = pd.read_csv(data)
    df = original_df
except ValueError:
    pass

if data:

    features = list(df)
    st.subheader('Your Data')
    data_showing = st.empty()

    data_showing.dataframe(df)
    revert_changes = st.sidebar.button('Revert data to initial state')

    if revert_changes:
        df = original_df
        data_showing.dataframe(df)

    clf_or_reg = st.sidebar.radio(
        "Classification or Regression",
        ('Classification', 'Regression'))

    drop_na = st.sidebar.checkbox('Remove rows with missing values')

    if drop_na:
        df = df.dropna()

    target = st.sidebar.multiselect('Select target', features)
    other_features = st.sidebar.multiselect('Select comparison feature', features)

    if target and other_features:
        st.subheader('Now showing cat-plots')
        for feature in other_features:
            st.pyplot(sns.catplot(x=feature, y=target[0], data=df).set_xticklabels(rotation=90))

    if target and clf_or_reg == 'Regression':
        cor = df.corr()
        cor_target = cor[target[0]]
        st.subheader('Pearson Correlation of features with the selected target')
        st.bar_chart(cor_target.drop(target[0]))

    features_to_remove = st.sidebar.multiselect('Select features to remove', features, key=31)

    remove_features = st.sidebar.button('Remove selected features from the data', on_click=remove_features())

    st.sidebar.markdown("***")

    data_type = df.dtypes
    categorical_features = data_type[(data_type == 'object') | (data_type == 'category')].index.tolist()

    features_to_encode = st.sidebar.multiselect('Select categorical features to one hot encode',
                                                categorical_features)

    encode = st.sidebar.button('encode!')

    if features_to_encode and encode:
        for feature in features_to_encode:
            df = encode_and_bind(df, feature)

        data_showing.dataframe(df)

    features_to_drop = set(categorical_features)
    features_to_drop.add(target[0])
    x = df.drop(features_to_drop, axis=1)
    y = df[target]

    split_amount = st.sidebar.slider("Select the train-test split ratio", 1, 100)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 - split_amount / 100), random_state=0)

    if target:
        st.sidebar.subheader('Would you like some PCA?')
        pca = PCA()
        pca.fit_transform(x_train)
        evr = pca.explained_variance_ratio_

        no_of_features = st.sidebar.number_input('Select number of features to include', 1, len(evr))

        st.sidebar.metric("Explained variance with %s features" % no_of_features,
                          "%" + "%.2f" % (sum(evr[0:no_of_features]) * 100))

        use_reduced_data = st.sidebar.checkbox('Use data with %s features' % no_of_features)

        if use_reduced_data:
            pca = PCA(n_components=no_of_features)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

    st.subheader('Let\'s Start Training')

    if clf_or_reg == 'Classification':
        classifier = st.selectbox('Select a classifier', classifiers)
        chosen_model = classifiers[classifier]['model']
        st.write(classifiers[classifier]['info'])

    if clf_or_reg == 'Regression':
        regressor = st.selectbox('Select a regressor', regressors)
        chosen_model = regressors[regressor]['model']
        st.write(regressors[regressor]['info'])

    begin_training = st.button('Train!')

    if begin_training and clf_or_reg == 'Classification':
        with st.spinner('Training your model...'):
            chosen_model.fit(x_train, y_train)
        st.success('Done! Your model is ready!')
        y_pred = chosen_model.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        accuracy = "%" + "%.2f" % (report.pop('accuracy') * 100)

        st.subheader('Classification Report')
        st.table(pd.DataFrame.from_dict(report).transpose())
        st.subheader('Accuracy Score : ' + accuracy)

        cm_dtc = confusion_matrix(y_test, y_pred)

        st.write('Confusion matrix: ', cm_dtc)

    elif begin_training:
        with st.spinner('Training your model...'):
            chosen_model.fit(x_train, y_train)
        st.success('Done! Your model is ready!')
        y_pred = chosen_model.predict(x_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", "%.2f" % r2_score(y_test, y_pred))
        col2.metric("Mean Squared Error", "%.2f" % mean_squared_error(y_test, y_pred))
        col3.metric("Mean Absolute Error", "%.2f" % mean_absolute_error(y_test, y_pred))

    elif begin_training and not chosen_model:
        st.write('Please choose a model first!')

    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='my-data.csv',
        mime='text/csv',
    )

else:
    st.sidebar.subheader('Start by uploading your data')
