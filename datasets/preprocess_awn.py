import pandas as pd
import numpy as np
import os
np.set_printoptions(threshold=np.inf)


features = [
        # ' LATITUDE_DEGREE',
        # ' LONGITUDE_DEGREE', 
        # ' ELEVATION_FEET', 
        ' MIN_AIR_TEMP', 
        ' AVG_AIR_TEMP', 
        ' MAX_AIR_TEMP', 
        ' MIN_HUMIDITY', 
        ' AVG_HUMIDITY', 
        ' MAX_HUMIDITY', 
        ' MIN_DEWPOINT', 
        ' AVG_DEWPOINT', 
        ' MAX_DEWPOINT', 
        ' SUM_PRECIP', 
        ' AVG_WIND_SP', 
        ' MAX_WIND_SP_MAX', 
        ' AVG_WIND_DIR', # authentic feature
        ' AVG_LEAF_WET', 
        ' SUM_SOLAR_RAD', 
        # ' MIN_SL_TEMP_2_IN', 
        # ' AVG_SL_TEMP_2_IN', 
        # ' MAX_SL_TEMP_2_IN', 
        ' MIN_SL_TEMP_8_IN', 
        ' AVG_SL_TEMP_8_IN', 
        ' MAX_SL_TEMP_8_IN', 
        # ' AVG_SL_MOIS_8_IN', 
        # ' AVG_SL_WP_8_IN', 
        # ' AVG_AIR_PRESSURE', 
        ' ETO', 
        ' ETR'
    ]

feats_hour = [
        # 'LATITUDE', 
        # 'LONGITUDE', 
        # 'ELEVATION_FEET', 
        'MIN_AIR_TEMP_F',
        'AIR_TEMP_F',
        'MAX_AIR_TEMP_F',
        'SECOND_AIR_TEMP_F',
        # 'AIR_TEMP_10M_F', 
        'RELATIVE_HUMIDITY_%',
        'DEWPOINT_F',
        # 'LEAF_WETNESS', 
        'PRECIP_INCHES',
        'SECOND_PRECIP_INCHES',
        'WIND_DIRECTION_2M_DEG',
        'WIND_SPEED_2M_MPH',
        'WIND_SPEED_MAX_2M_MPH',
        # 'WIND_DIRECTION_10M_DEG', 
        # 'WIND_SPEED_10M_MPH', 
        # 'WIND_SPEED_MAX_10M_MPH', 
        'SOLAR_RAD_WM2',
        'SOIL_TEMP_2_IN_DEGREES_F', 
        'SOIL_TEMP_8_IN_DEGREES_F',
        # 'SOIL_WP_2_IN_KPA', 
        # 'SOIL_WP_8_IN_KPA', 
        'SOIL_MOIS_8_IN_%'
    ]
def make_data():
    awn_folder_name = 'Daily'
    m_dict = {
        'Daily': 1,
        'Hourly': 24,
        'Minutes15': 24 * 4
    }
    m = m_dict[awn_folder_name]
    data_folder = './data'
    awn_folder = f'{data_folder}/{awn_folder_name}'
    awn_list = ['AWN_DAILY_110029.csv'] #['AWN_DAILY_330127.csv'] # #os.listdir(awn_folder)
    is_year = True
    data = []
    if is_year:
        length = 366
        for filename in awn_list: # os.listdir(awn_folder):
            if filename.endswith('.csv'):
                df = pd.read_csv(f'{awn_folder}/{filename}')
                # df = df.replace("NA", np.nan)
                # print(f"{df}")
                # print(f"{df[' STATION_NAME'].unique()}, filename: {filename}")
                X = []
                x = []
                year = ''
                for i in range(len(df)):
                    yy = df.iloc[i][' JULDATE'].split('-')[0]
                    if i != len(df) -1 and (year == '' or year == yy):
                        x.append(df.iloc[i][features].to_list())
                        year = yy
                    else:

                        year = yy
                        if i == len(df) - 1:
                            x.append(df.iloc[i][features].to_list())
                        if len(x) < length:
                            for i in range(length - len(x)):
                                x.append([0 for j in range(len(features))])
                        X.append(np.array(x))
                        x = []
                        if i != len(df) - 1:
                            x.append(df.iloc[i][features].to_list())
                            
                X = np.array(X)
                if len(X.shape) == 3:
                    data.extend(X.copy())
                    total = X.shape[0] * X.shape[1] * X.shape[2]
                    miss = np.isnan(X).sum()
                    print(f"filename: {filename}: X = {X.shape}\ntotal: {total}, miss: {miss}, ratio: {total / miss if miss != 0 else -1}")

    else:
        for filename in os.listdir(awn_folder):
            if filename.endswith('.csv'):
                df = pd.read_csv(f'{awn_folder}/{filename}')
                month = ''
                X = []
                x = []
                for i in range(len(df)):
                    if awn_folder_name == 'Daily':
                        mm = df.iloc[i][' JULDATE'].split('-')[1]
                    elif awn_folder_name == 'Hourly':
                        mm = df.iloc[i]['TSTAMP_PST'].split('-')[1]

                    if month == '':
                        month = mm
                        if awn_folder_name == 'Daily':
                            x.append(df.iloc[i][features].to_list())
                        elif awn_folder_name == 'Hourly':
                            x.append(df.iloc[i][feats_hour].to_list())
                    elif month == mm:
                        if awn_folder_name == 'Daily':
                            x.append(df.iloc[i][features].to_list())
                        elif awn_folder_name == 'Hourly':
                            x.append(df.iloc[i][feats_hour].to_list())
                    else:
                        month = mm
                        if len(x) < m * 31:
                            for i in range(m * 31 - len(x)):
                                if awn_folder_name == 'Daily':
                                    x.append([0 for j in range(len(features))])
                                elif awn_folder_name == 'Hourly':
                                    x.append([0 for j in range(len(feats_hour))])
                        X.append(np.array(x))
                        x = []
                X = np.array(X)
                data.extend(X)

    data = np.array(data)
    print(f'data: {data.shape}')

    np.save(f'{awn_folder}/data_yy.npy', data)
    # print(data)
    miss_data = []
    total_nan = 0
    for i in range(len(data)):
        miss = np.isnan(data[i]).sum()
        # print(f"miss = {miss}")
        total_nan += miss
        if miss > 0:
            miss_data.append(data[i])

    print(f"Total missing: {total_nan}")
    miss_data = np.array(miss_data)
    print(f"Miss data: {miss_data.shape}")
    print(f"Total number: {miss_data.shape[0] * miss_data.shape[1] * miss_data.shape[2]}")
    np.save(f'{awn_folder}/miss_data_yy.npy', miss_data)

# make_data()