import pandas as pd
import numpy as np
from pycaret.regression import *
import streamlit as st


#loading model
v3 = load_model('v3_model')





# Streamlit UI
left1, right1 = st.columns(2, gap = "large")

with left1:
    au = st.number_input("Orbit Semi-Major Axis (AU)", 
                        min_value=0.01,
                        max_value=50.0,
                        value=1.0,
                        step=0.01,
                        help="Average Distance Of The Planet From Its Star")
    sl = st.number_input("Stellar Effective Temperature (K)", 
                        min_value=1000,
                        max_value=10000,
                        value=5800,
                        step=50,
                        help="Surface Temperature Of The Host Star")
    unit = st.selectbox("Choose temperature unit:", ["Kelvin (K)", "Celsius (°C)", "Fahrenheit (°F)"])


F1 = sl
F2 = sl * (np.sqrt(1 / (2*au))) * ((1 - 0.3) ** 0.25)
#F3 = sl * np.sqrt(1 / (2*au)) * ((1 - 0.3) ** 0.25)
F4 = np.log10(sl)
F5 = np.log10(1 / (2*au))
F6 = np.log10(sl * np.sqrt(1 / (2*au)) * ((1 - 0.3) ** 0.25))



input_data = pd.DataFrame([[F1, F2, F4, F5, F6]], columns=['feat1', 'feat2', 'feat4', 'feat5', 'feat6'])

if st.button("Predict Equilibrium Temperature"):
        
        with right1:
            output = predict_model(v3, data = input_data)
            y_pred = output.loc[0, 'prediction_label']
            result = float(f"{y_pred:.2f}")


            #Deciding unit
            def decide_unit(result, unit):
                if unit == "Celsius (°C)":
                    result = float(f"{y_pred - 273.15:.2f}")
                    u = "°C"
                elif unit == "Fahrenheit (°F)":
                    result = float(f"{((y_pred - 273.15) * 9/5 + 32):.2f}")
                    u = "°F"
                else:
                    result = float(f"{y_pred:.2f}")
                    u = "K"
                return result, u
            
            #CLassifying the planet type
            tags = []
            if int(result) > 230 and int(result) < 350:
                result, u = decide_unit(result, unit)
                color = "#3CB371"  
                emoji = "🌎"
                display_val = str(result) 
                tags = ["Temperate", "Earth-like"]
                type = f":green-background[{tags[0]}] :green-background[{tags[1]}]"
            elif int(result) >= 350:
                result, u = decide_unit(result, unit)
                color = "#FF6347"  
                emoji = "🔥"
                display_val = str(result) 
                tags = ["Hot"]
                type = f":red-background[{tags[0]}]"
            elif(int(result) <= 230):
                result, u = decide_unit(result, unit)
                color = "#1E90FF"  
                emoji = "❄️"
                display_val = str(result) 
                tags = ["Cold"]
                type = f":blue-background[{tags[0]}]"

            

            st.markdown(f"""
                <div style="position: relative; margin-top: 0px;">

                <!-- Background Earth Emoji -->
                <span style="
                font-size: 120px;
                opacity: 0.13;
                position: absolute;
                top: -35px;
                left: 45%;
                transform: translateX(-50%);
                z-index: 0;
                ">{emoji}</span>

                <h1 style="position: relative; z-index: 1;">
                <b style='color: {color}'>{display_val}</b>
                <span style='font-size: 0.7em'><i>{u}</i></span>
                </h1>

                </div>
                """, unsafe_allow_html=True)
            st.write(type)
            #st.header(f":MediumAquaMarine[{display_val}]")
            
