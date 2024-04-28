import pandas as pd
import numpy as np
import os
import re
import dask.array as da
np.set_printoptions(threshold=np.inf)

folder = "./data/AWN"

m = 96 * 7

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


index = 0
X = []
out_folder = "./data/AWN/train"
count_idx = 0
for file in os.listdir(folder):
    
    if file.endswith('.csv'):
        print(f"file: {file}")
        df = pd.read_csv(f"{folder}/{file}", dtype={
            'TSTAMP_PST': str, 
            ' UNIT_ID': int, 
            ' STATION_NAME': str, 
            ' LATITUDE': float, 
            ' LONGITUDE': float, 
            ' ELEVATION_FEET': float, 
            ' AIR_TEMP_F': float, 
            ' SECOND_AIR_TEMP_F': float, 
            'AIR_TEMP_10M_F': float, 
            ' RELATIVE_HUMIDITY_%': float, 
            ' DEWPOINT_F': float, 
            ' LEAF_WETNESS': float, 
            ' PRECIP_INCHES': float, 
            ' SECOND_PRECIP_INCHES': float, 
            ' WIND_DIRECTION_2M_DEG': float, 
            ' WIND_SPEED_2M_MPH': float, 
            ' WIND_SPEED_MAX_2M_MPH': float, 
            ' WIND_DIRECTION_10M_DEG': float, 
            ' WIND_SPEED_10M_MPH': float, 
            ' WIND_SPEED_MAX_10M_MPH': float, 
            ' SOLAR_RAD_WM2': float, 
            ' SOIL_TEMP_2_IN_DEGREES_F': float, 
            '  SOIL_TEMP_8_IN_DEGREES_F': float, 
            '  SOIL_WP_2_IN_KPA': float, 
            '  SOIL_WP_8_IN_KPA': float, 
            '  SOIL_MOIS_8_IN_%': float
        }) 
        x = []
        count = 0
        for i in range(len(df)):
            count+=1
            if count == m:
                X.append(np.stack(x))
                x = []
                count = 0
            else:
                x.append(df.iloc[i][features])
        if len(x) < m:
            adds = m - len(x)
            
            for i in range(len(adds)):
                y = []
                for j in range(len(x[0])):
                    y.append(np.nan)
                x.append(y)
            X.append(np.stack(x))
            x = []
            count = 0
        index += 1
        if index % 60 == 0:
            X = np.array(X)
            print(f"X: {X.shape}")
            
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            np.save(f"{out_folder}/X_train_{count_idx}.npy", X)
            count_idx += 1
            X = []
if index % 60 != 0:
    X = np.array(X)
    print(f"X: {X.shape}")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    np.save(f"{out_folder}/X_train_{count_idx}.npy", X)

            