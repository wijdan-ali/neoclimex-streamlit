import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



#Training the model

sample = pd.read_csv("raw_data/exo_teq_results.csv")

T_star = sample["st_teff"].values
R_star = sample["st_rad"].values
a_planet = sample["pl_orbsmax"].values

#Engineering Features
Feat1 = T_star.copy()
Feat2 = np.sqrt(1 / a_planet)
#Feat2 = np.sqrt(R_star / a_planet)
Feat3 = T_star * np.sqrt(1 / a_planet)
Feat4 = np.log10(T_star)
Feat5 = np.log10(1 / a_planet)

data_train = pd.DataFrame({
    "feat1": Feat1,
    "feat2": Feat2,
    "feat3": Feat3,
    "feat4": Feat4,
    "feat5": Feat5,
    #"feat6": Feat6,
    "Teq": sample["pl_eqt"],
    "type": sample["type"]
})

#hot_test = data_train[data_train["type"] == "hot"].sample(35)
#hotter_test = data_train[data_train["type"] == "hotter"].sample(25)
#cold_test = data_train[data_train["type"] == "cold"].sample(5)
#temperate_test = data_train[data_train["type"] == "temperate"].sample(15)

#data_test = pd.concat([hot_test, hotter_test, cold_test, temperate_test], ignore_index = True)
#data_train = data_train.drop(data_test.index)

x_train = data_train[["feat1", "feat2", "feat3", "feat4", "feat5"]].values
y_train = data_train["Teq"].values
#x_test = data_test[["feat1", "feat2","feat3", "feat4", "feat5"]].values
#y_test = data_test["Teq"].values

linreg = LinearRegression()
linreg.fit(x_train, y_train)



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
F2 = np.sqrt(1 / au)
F3 = sl * np.sqrt(1 / au)
F4 = np.log10(sl)
F5 = np.log10(1 / au)


input_data = np.array([[F1, F2, F3, F4, F5]])
if st.button("Predict Equilibrium Temperature"):
        
        with right1:
            y_pred = linreg.predict(input_data)
            result = float(f"{y_pred[0]:.2f}")


            #Deciding unit
            def decide_unit(result, unit):
                if unit == "Celsius (°C)":
                    result = float(f"{y_pred[0] - 273.15:.2f}")
                    u = "°C"
                elif unit == "Fahrenheit (°F)":
                    result = float(f"{((y_pred[0] - 273.15) * 9/5 + 32):.2f}")
                    u = "°F"
                else:
                    result = float(f"{y_pred[0]:.2f}")
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
                tags = ["Hot Jupiter"]
                type = f":red-background[{tags[0]}]"
            elif(int(result) <= 230):
                result, u = decide_unit(result, unit)
                color = "#1E90FF"  
                emoji = "❄️"
                display_val = str(result) 
                tags = ["Cold Neptune"]
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
            


    







#st.write(f"Test RMSE: {rmse:.2f}")
#st.write(f"Test R^2 Score: {r2:.3f}")
    