import streamlit as st
import pickle
import numpy as np
import pandas as pd

# load model and transformer
model = pickle.load(open('model/model.pkl', 'rb'))
transformer = pickle.load(open('model/transformer.pkl', 'rb'))

locations = ['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli', 'Lingadheeranahalli', 'Kothanur', 'Whitefield', 'Old Airport Road', 'Marathahalli', '7th Phase JP Nagar', 'Gottigere', 'Sarjapur', 'Mysore Road', 'Bisuvanahalli', 'Raja Rajeshwari Nagar', 'other', 'Binny Pete', 'Thanisandra', 'Bellandur', 'Electronic City', 'Ramagondanahalli', 'Yelahanka', 'Hebbal', 'Kasturi Nagar', 'Kanakpura Road', 'Electronics City Phase 1', 'Kundalahalli', 'Chikkalasandra', 'Murugeshpalya', 'Sarjapur  Road', 'Doddathoguru', 'KR Puram', 'Bhoganhalli', 'Lakshminarayana Pura', 'Begur Road', 'Devanahalli', 'Varthur', 'Bommanahalli', 'Gunjur', 'Hegde Nagar', 'Haralur Road', 'Hennur Road', 'Kothannur', 'Kalena Agrahara', 'Garudachar Palya', 'EPIP Zone', 'Dasanapura', 'Kasavanhalli', 'Sanjay nagar', 'Domlur', 'Kengeri', 'Sarjapura - Attibele Road', 'Yeshwanthpur', 'Nagarbhavi', 'Rajaji Nagar', 'Ramamurthy Nagar', 'Malleshwaram', 'Akshaya Nagar', 'Kadugodi', 'LB Shastri Nagar', 'Hormavu', 'Kudlu Gate', '8th Phase JP Nagar', 'Bommasandra Industrial Area', 'Anandapura', 'Kengeri Satellite Town', 'Kannamangala', ' Devarachikkanahalli', 'Hulimavu', 'Hosa Road', 'CV Raman Nagar', 'Hebbal Kempapura', 'Vijayanagar', 'Pattandur Agrahara', 'HSR Layout', 'Kogilu', 'Panathur', 'Padmanabhanagar', '1st Block Jayanagar', 'Kammasandra', 'Dasarahalli', 'Magadi Road', 'Koramangala', 'Budigere', 'Kalyan nagar', 'OMBR Layout', 'Horamavu Agara', 'Ambedkar Nagar', 'Attibele', 'Talaghattapura', 'Balagere', 'Jigani', 'Gollarapalya Hosahalli', 'Old Madras Road', 'Kaggadasapura', 'Chandapura', '9th Phase JP Nagar', 'Jakkur', 'TC Palaya', 'Singasandra', 'AECS Layout', 'Mallasandra', 'Begur', 'JP Nagar', 'Malleshpalya', 'Munnekollal', 'Giri Nagar', 'Kaval Byrasandra', 'Kaggalipura', '6th Phase JP Nagar', 'Ulsoor', 'Thigalarapalya', 'Somasundara Palya', 'Basaveshwara Nagar', 'Bommasandra', 'Ardendale', 'Harlur', 'Kodihalli', 'Narayanapura', 'Hennur', '5th Phase JP Nagar', 'Kodigehaali', 'Bannerghatta Road', 'Billekahalli', 'Jalahalli', 'Mahadevpura', 'Anekal', 'Sompura', 'Dodda Nekkundi', 'Hosur Road', 'Battarahalli', 'Ambalipura', 'Hoodi', 'Brookefield', 'Yelenahalli', 'Vittasandra', 'Vidyaranyapura', 'Amruthahalli', 'Subramanyapura', 'Kereguddadahalli', 'Kambipura', 'Sector 7 HSR Layout', 'Rajiv Nagar', 'Arekere', 'Mico Layout', 'Kammanahalli', 'Banashankari', 'Chikkabanavar', 'HRBR Layout', 'Banashankari Stage III', 'Nehru Nagar', 'Kanakapura', 'Konanakunte', 'R.T. Nagar', 'Tumkur Road', 'GM Palaya', 'Jalahalli East', 'Hosakerehalli', 'Indira Nagar', 'Kodichikkanahalli', 'Varthur Road', 'Anjanapura', 'Abbigere', 'Dommasandra', 'Gubbalala', 'Dairy Circle', 'Kudlu', 'Banashankari Stage VI', 'Cox Town', 'Kathriguppe', 'Yelahanka New Town', 'Sahakara Nagar', 'Rachenahalli', 'Yelachenahalli', 'Green Glen Layout', 'Thubarahalli', 'Horamavu Banaswadi', '1st Phase JP Nagar', 'NGR Layout', 'Seegehalli', 'NRI Layout', 'ITPL', 'Margondanahalli', 'Babusapalaya', 'Nagappa Reddy Layout', 'Iblur Village', 'Channasandra', 'Choodasandra', 'Kaikondrahalli', 'Neeladri Nagar', 'Frazer Town', 'Cooke Town', 'Doddakallasandra', 'Chamrajpet', 'Rayasandra', 'Kalkere', 'Pai Layout', 'Banashankari Stage V', 'Sonnenahalli', 'Benson Town', 'Poorna Pragna Layout', 'Judicial Layout', 'Banashankari Stage II', 'Karuna Nagar', 'Ananth Nagar', 'Basavangudi', 'Bannerghatta', 'Bommenahalli', 'HBR Layout', 'Laggere', 'Prithvi Layout', 'Banaswadi', 'Sector 2 HSR Layout', 'Nagavarapalya', 'BTM Layout', 'Nagavara', 'BTM 2nd Stage', '1st Block Koramangala', 'Hoskote', 'Doddaballapur', 'Kumaraswami Layout', 'Gunjur Palya', 'Sarakki Nagar', 'Bharathi Nagar', 'Sultan Palaya', 'Kadubeesanahalli', 'Shivaji Nagar', 'Kenchenahalli', 'Cunningham Road']

st.title("Bengaluru House Price Predictor")
with st.expander("ℹ️ What do these terms mean?"):
    st.markdown("""
    **Area Type**
    - **Super built-up Area** — What builders advertise. Includes your flat + shared spaces like lift, staircase, lobby. Most listings use this.
    - **Built-up Area** — Your flat's space including walls and balcony. No shared spaces.
    - **Carpet Area** — The actual floor space inside your 4 walls. Smallest number, most honest.
    - **Plot Area** — Only for independent houses/bungalows where you own the land. Not for flats.
    
    **BHK** — Number of Bedrooms, Hall and Kitchen. A 2BHK has 2 bedrooms + 1 hall + 1 kitchen.
    """)

location = st.selectbox("Location", sorted(locations))
area_type = st.selectbox("Area Type", 
    ['Super built-up  Area', 'Built-up  Area', 'Plot  Area', 'Carpet  Area'],
    help="Super built-up includes common areas. Built-up is wall to wall. Carpet is usable area inside walls. Plot is land area.")
prop_type = 'BHK'
total_sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1000)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Balconies", min_value=0, max_value=3, value=1)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area_type, location, prop_type, total_sqft, bath, balcony, bhk]],
                               columns=['area_type', 'location', 'type', 'total_sqft', 'bath', 'balcony', 'bhk'])
    
    transformed = transformer.transform(input_data)
    log_price = model.predict(transformed)
    price = np.expm1(log_price[0])
    
    st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")