import pandas as pd
import numpy as np
import os
import re
np.set_printoptions(threshold=np.inf)

folder = "/Users/rai009/Documents/qual_fresh/AgWeatherNet/Paolo"

m = 96 * 31

features = [
    # 'TSTAMP_PST', 
    # ' UNIT_ID', 
    # ' STATION_NAME', 
    # ' LATITUDE', 
    # ' LONGITUDE', 
    # ' ELEVATION_FEET', 
    ' AIR_TEMP_F', 
    ' SECOND_AIR_TEMP_F', 
    'AIR_TEMP_10M_F', 
    ' RELATIVE_HUMIDITY_%', 
    ' DEWPOINT_F', 
    ' LEAF_WETNESS', 
    ' PRECIP_INCHES', 
    ' SECOND_PRECIP_INCHES', 
    ' WIND_DIRECTION_2M_DEG', 
    ' WIND_SPEED_2M_MPH', 
    ' WIND_SPEED_MAX_2M_MPH', 
    ' WIND_DIRECTION_10M_DEG', 
    ' WIND_SPEED_10M_MPH', 
    ' WIND_SPEED_MAX_10M_MPH', 
    ' SOLAR_RAD_WM2', 
    ' SOIL_TEMP_2_IN_DEGREES_F', 
    '  SOIL_TEMP_8_IN_DEGREES_F', 
    '  SOIL_WP_2_IN_KPA', 
    '  SOIL_WP_8_IN_KPA', 
    '  SOIL_MOIS_8_IN_%'
]


X = []
for file in os.listdir(folder):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(f"{folder}/{file}") 
            x = []
            count = 0
            for i in range(len(df)):
                count+=1
                if count == m:
                    X.append(x)
                    x = []
                    count = 0
                else:
                    x.append(df.iloc[i][features])
        except:
            continue
X = np.array(X)
print(f"X: {X.shape}")
out_folder = "./data/AWN"
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)
np.save(f"{out_folder}/X_train.npy", X)

            