import streamlit as st
from streamlit_option_menu import option_menu
import joblib
from joblib import load
import numpy as np
import folium
from streamlit_folium import folium_static

# Add a title to your Streamlit app and page configuration

st.set_page_config(page_title= "Singapore Resale Flat Price Predictor",
                   layout= "wide",initial_sidebar_state='expanded')
st.markdown('<h1 style="color:#fbe337;text-align: center;">Singapore Resale Flat Price Predictor</h1>', unsafe_allow_html=True)

# Set up the option menu

menu=option_menu("",options=["Project Overview","Flat Resale Price Prediction"],
                        icons=["house","cash"],
                        default_index=1,
                        orientation="horizontal",
                        styles={
        "container": {"width": "100%", "border": "2px ridge", "background-color": "#333333"},
        "icon": {"color": "#FFD700", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#FFFFFF"},
        "nav-link-selected": {"background-color": "#555555", "color": "#FFFFFF"}})

# set up the information for 'Project Overview' menu
if menu == "Project Overview":
    # Project Title
    st.subheader(':orange[Project Title:]')
    st.markdown('<h5>Singapore Resale Flat Prices Predicting</h5>', unsafe_allow_html=True)
    
    # Problem Statement
    st.subheader(':orange[Problem Statement:]')
    st.markdown('''<h5>The objective of this project is to develop a machine learning model and 
                deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. 
                This predictive model will be based on historical data of resale flat transactions,
                and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.</h5>''', 
                unsafe_allow_html=True)
    
    # Scope
    st.subheader(':orange[Scope:]')
    st.markdown('''<h5>
                - Data Collection and Preprocessing<br>
                - Feature Engineering<br>
                - Model Selection and Training<br>
                - Model Evaluation<br>
                - Streamlit Web Application<br>
                - Deployment on Render<br>
                - Testing and Validation
                </h5>''', unsafe_allow_html=True)
    
    # Technologies and Tools
    st.subheader(':orange[Technologies and Tools:]')
    st.markdown('<h5>Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Streamlit</h5>', unsafe_allow_html=True)
    
    # Project Learning Outcomes
    st.subheader(':orange[Project Learning Outcomes:]')
    st.markdown('''<h5>
                - Understanding of the end-to-end process of developing a predictive model<br>
                - Practical experience with data collection, preprocessing, and feature engineering<br>
                - Knowledge of various machine learning algorithms and their applications<br>
                - Skills in evaluating model performance and tuning hyperparameters<br>
                - Experience in deploying machine learning models as web applications<br>
                - Familiarity with Streamlit for building interactive web applications<br>
                - Hands-on experience with deploying applications on Render<br>
                </h5>''', unsafe_allow_html=True)
    
    # Additional Suggested Headers
    st.subheader(':orange[Data Sources:]')
    st.markdown('<h5>Publicly available datasets from Singapore government agencies such as HDB and Data.gov.sg</h5>', unsafe_allow_html=True)
    
    st.subheader(':orange[Target Audience:]')
    st.markdown('<h5>Potential buyers, sellers, real estate agents, and analysts interested in the Singapore real estate market.</h5>', unsafe_allow_html=True)

# town coordinates
town_coordinates = {
    'ANG MO KIO': [1.3691, 103.8454],'BEDOK': [1.3244, 103.9305],'BISHAN': [1.3572, 103.8511],'BUKIT BATOK': [1.3561, 103.7582],
    'BUKIT MERAH': [1.2780, 103.8126],'BUKIT TIMAH': [1.3302, 103.7725],'CENTRAL AREA': [1.2903, 103.8510],'CHOA CHU KANG': [1.3790, 103.7420],
    'CLEMENTI': [1.3174, 103.7635],'GEYLANG': [1.3133, 103.8738],'HOUGANG': [1.3662, 103.8850],'JURONG EAST': [1.3334, 103.7400],
    'JURONG WEST': [1.3491, 103.6942],'KALLANG/WHAMPOA': [1.3140, 103.8687],'MARINE PARADE': [1.3038, 103.9015],'QUEENSTOWN': [1.2870, 103.7960],
    'SENGKANG': [1.3874, 103.8950],'SERANGOON': [1.3592, 103.8688],'TAMPINES': [1.3530, 103.9450],'TOA PAYOH': [1.3337, 103.8486],
    'WOODLANDS': [1.4424, 103.7886],'YISHUN': [1.4275, 103.8378],'LIM CHU KANG': [1.4296, 103.6983],'SEMBAWANG': [1.4451, 103.8180],
    'BUKIT PANJANG': [1.3781, 103.7661],'PASIR RIS': [1.3718, 103.9446],'PUNGGOL': [1.4042, 103.9155]}

# User input Values:
class columns():

    town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 
          'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 
          'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    
    town_encoded={'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4, 'BUKIT TIMAH': 6, 'CENTRAL AREA': 7,
                'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11, 'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14,
                'MARINE PARADE': 16, 'QUEENSTOWN': 19, 'SENGKANG': 21, 'SERANGOON': 22, 'TAMPINES': 23, 'TOA PAYOH': 24, 'WOODLANDS': 25,
                'YISHUN': 26, 'LIM CHU KANG': 15, 'SEMBAWANG': 20, 'BUKIT PANJANG': 5, 'PASIR RIS': 17, 'PUNGGOL': 18}
    
    flat_type=['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']

    flat_type_encoded={'1 ROOM': 0, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, '2 ROOM': 1, 'EXECUTIVE': 5, 'MULTI GENERATION': 6}

    flat_model=['2-ROOM', '3GEN', 'ADJOINED FLAT', 'APARTMENT', 'DBSS', 'IMPROVED', 'IMPROVED-MAISONETTE', 'MAISONETTE', 'MODEL A', 'MODEL A-MAISONETTE', 
                       'MODEL A2', 'MULTI GENERATION', 'NEW GENERATION', 'PREMIUM APARTMENT', 'PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE', 'SIMPLIFIED', 'STANDARD', 
                       'TERRACE', 'TYPE S1', 'TYPE S2']
    
    flat_model_encoded={'IMPROVED': 5, 'NEW GENERATION': 12, 'MODEL A': 8, 'STANDARD': 17, 'SIMPLIFIED': 16, 'MODEL A-MAISONETTE': 9, 'APARTMENT': 3,
                        'MAISONETTE': 7, 'TERRACE': 18, '2-ROOM': 0, 'IMPROVED-MAISONETTE': 6, 'MULTI GENERATION': 11, 'PREMIUM APARTMENT': 13, 
                        'ADJOINED FLAT': 2, 'PREMIUM MAISONETTE': 15, 'MODEL A2': 10, 'DBSS': 4, 'TYPE S1': 19, 'TYPE S2': 20, 'PREMIUM APARTMENT LOFT': 14, '3GEN': 1}

if menu == 'Flat Resale Price Prediction':
    st.markdown("<h4 style=color:#fbe337>Enter the following details:",unsafe_allow_html=True)
    st.write('')

    with st.form('price_prediction'):
        col1,col2=st.columns(2)

        with col1:
            Year=st.slider('**Year**', min_value=1990, max_value=2024)
            Flat_Type=st.selectbox('**Flat Type**',columns.flat_type,index=None)
            Town=st.selectbox(label='**Town**',options=columns.town,index=None)
            Storey_initial=st.slider('**Storey_initial**', min_value=1, max_value=100)

        with col2:
            Lease_Commence_Year=st.slider('**Lease_Commence_Year**', min_value=1966, max_value=2025)
            Flat_Model=st.selectbox('**Flat Model**',columns.flat_model,index=None)
            Floor_Area_sqm=st.number_input('**Floor Area (sqm)**',min_value=0.1)
            Storey_end=st.slider('**Storey_end**', min_value=1, max_value=100)

        st.write("")

        col1,col2,col3 = st.columns([3,5,3])
        with col2:

            button=st.form_submit_button(':orange[**Predict Flat Resale Price**]',use_container_width=True)

        if button:
            if not all([Year,Flat_Type,Town,Storey_initial,Lease_Commence_Year,Flat_Model,Floor_Area_sqm,Storey_end]):
                st.error('Please fill in all the required fields.')
            else:
                new_df = load("Resale_flat_price_model.joblib", mmap_mode='r')


                # Encode the input values
                Town_encoded= columns.town_encoded[Town]
                Flat_Type_encoded= columns.flat_type_encoded[Flat_Type]
                Flat_Model_encoded= columns.flat_model_encoded[Flat_Model]

            #predict the status with regressor model
            input_data=np.array([[Year,Flat_Type_encoded,Town_encoded,Storey_initial,Lease_Commence_Year,Flat_Model_encoded,Floor_Area_sqm,Storey_end]])

            predict=new_df.predict(input_data)
            predicted_resale_price = predict[0]
            st.success(f"Predicted Resale Price: {predicted_resale_price:.2f}")
            st.snow()



            # Display the map with a marker for the selected town

            map_center = [1.3521, 103.8198]  # Coordinates for Singapore
            m = folium.Map(location=map_center, zoom_start=12)
            if Town in town_coordinates:
                folium.Marker(town_coordinates[Town], popup=Town).add_to(m)
            folium_static(m)









                