import streamlit as st
import pandas as pd

from Utils import extract_timeseries_features, validate_utc_timestamp_column, timestamp_getter
from Utils import load_json, scaling_model, pca_model, trained_model
from Utils import ID_COLUMN_NAME


ID = 1
COLUMNS_TO_DROP = ['friedrich_coefficients__coeff_0__m_3__r_30', 'friedrich_coefficients__coeff_1__m_3__r_30', 'friedrich_coefficients__coeff_2__m_3__r_30', 'friedrich_coefficients__coeff_3__m_3__r_30', 'max_langevin_fixed_point__m_3__r_30', 'query_similarity_count__query_None__threshold_0.0']
LABEL_MAPPING = {
    "0": "ADHD Negative",
    "1": "ADHD Positive"
}
IMAGE_ADDRESS = "https://www.health.com/thmb/ErL7FfqsBqpS2K5cXMuLb_X-hjQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/Health-adhd-overview-Mira-Norian-final-3d6ecb1ef018467ab60b9b93158edeba.jpg"

# load json data
json_columns = load_json()
# load the scaling model
min_max_scaler = scaling_model()
# load pca model
pca_dim_reducer = pca_model()
# load pred model
pred_model = trained_model()

# Title of the app
st.title('ADHD Detection')

# set the image
st.image(IMAGE_ADDRESS, caption = "ADHD_Detection")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    start_time  = timestamp_getter()
    # Read the file to a dataframe using pandas
    df = pd.read_csv(uploaded_file)

    # show the dataframe, if needs
    with st.sidebar:
        st.subheader("User Data")
        # Display the dataframe
        st.write(df)

    # get the columns
    columns = list(df.columns)

    # create a drop down
    timestamp_column = st.selectbox("Seelct the timestamp column", columns, index = None)

    if not timestamp_column:
        st.warning('Please select the timestamp column', icon="⚠️")
        st.stop()
    
    # process columns
    leftover_column = [col for col in columns if col != timestamp_column]
    cols_to_drop = [f"{leftover_column[0]}__{column}" for column in COLUMNS_TO_DROP]

    # validate column
    timestamp_validation = validate_utc_timestamp_column(df, timestamp_column)

    # attach a ID column
    df[ID_COLUMN_NAME] = str(ID)

    with st.status("Making Predictions"):
        st.write("Start Feature Extraction")
        # extract the features
        feature_process = False
        try:
            extracted_data = extract_timeseries_features(df, timestamp_column)
            #st.success('Feature Extraction Successful!', icon="✅")
            feature_process = True
        except Exception as error:
            print(str(error))
            st.error('Cannot Extract the Features', icon="🚨")
        
        if not feature_process:
            st.error('Cannot Make Predictions!', icon="🚨")
            st.write("-  Check whether you have a **timestamp** column.")
            st.write("-  Check whether you have specified the correct **timestamp** column.")
            st.stop()
        st.write("Processing the Data")
        # drop unwanted columns
        extracted_data.drop(cols_to_drop, axis = 1, inplace = True)
        # drop ID column
        extracted_data.drop(ID_COLUMN_NAME, axis = 1, inplace =True)
        # drop uniform columns
        extracted_data.drop(json_columns["uniform_cols"], axis = 1, inplace = True)
        # start scaling
        extracted_data[json_columns["cols_to_scale"]] = min_max_scaler.transform(extracted_data[json_columns["cols_to_scale"]])
        # Processing dimensions
        pca_transformed_x = pca_dim_reducer.transform(extracted_data)
        # make predictions
        st.write("Getting Predictions")
        predictions = pred_model.predict(pca_transformed_x)
        st.toast('Predictions Completed!', icon="✅")

    # show the predictions
    st.subheader("User Condition: {}".format(LABEL_MAPPING[str(predictions[0])]))
    end_time = timestamp_getter()
    st.write("Predictions took: {} seconds".format(end_time - start_time))
    
else:
    st.info('Please upload a CSV file.')
