import numpy as np

feats_v1 = ['sin', 'cos2', 'tan']
feats_v2 = ['sin', 'cos', 'tan']
feats_v3 = ['sin', 'cos', 'sin-cos-plus', 'sin-cos-prod']
feats_v4 = ['sin', '1+sin2', 'cos', '1-cos2', 'tan', '1+tan']
feats_v5 = ['sin', '1+sin2', 'cos', '1-cos2', 'cos2-sin2']
feats_v6 = ['sin', 'cos', '1+sin2', 'tan']
feats_v7 = ['sin', '1+sin2', 'cos', '1-cos2', 'tan', '1+tan']
feats_v8 = ['sin', 'cos', 'tan']


def add_rn_missing(X, length=-1, rate=-1):
    if length != -1:
        a = np.arange(X.shape[0] - length + 1)
        start = np.random.choice(a)
        end = start + length + 1
        X[start:end] = np.nan
    else:
        shp = X.shape
        sample = X.reshape(-1).copy()
        indices = np.where(~np.isnan(sample))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        sample[indices] = np.nan
        sample = sample.reshape(shp)
        X = sample
    return X
    


def create_synthetic_data_v1(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)

    synth_features = {
        'sin': (0.00001, 2 * np.pi/3, np.pi/2), # f1
        'cos2': (0, 2*np.pi, np.pi/4), # f2
        'tan': (-np.pi/3, np.pi/3, np.pi/16) # f3
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v1) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))

    for i in range(data.shape[0]):
        for feature in feats_v1:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v1.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == 'cos2':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v1.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])) ** 2)
            elif feature == 'tan':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v1.index(feature)] = np.tan(np.linspace(low, high, data.shape[1]))

            if feature == 'sin' or feature == 'cos2' or feature == 'tan':
                if length_rate != 0:
                    data[i, :, feats_v1.index(feature)] = add_rn_missing(data[i, :, feats_v1.index(feature)], length=lr)
            
        if rate != 0:
            data[i] = add_rn_missing(data[i], rate=rate)
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std


def create_synthetic_data_v2(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False, is_mcar=False, is_col_miss=None):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/10), # f1
        'cos': (-np.pi/4, np.pi/3, np.pi/10), # f2
        'tan': (0) #, np.pi/3, np.pi/10) # f3
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v2) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in feats_v2:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v2.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
                
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v2.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == 'tan':
                f1_sin = data[i, :, feats_v2.index('sin')]
                f2_cos = data[i, :, feats_v2.index('cos')]
                data[i, :, feats_v2.index(feature)] = f1_sin / (f2_cos + 1e-5) #+ np.random.rand(data.shape[1])
            if noise and feature != 'tan':
                noise_level = np.random.rand(data.shape[1]) * 0.5
                data[i, :, feats_v2.index(feature)] += noise_level

            if not is_mcar and (feature == 'sin' or feature == 'cos'):
                if length_rate != 0:
                    data[i, :, feats_v2.index(feature)] = add_rn_missing(data[i, :, feats_v2.index(feature)], length=lr)
    
        if is_col_miss is not None:
            r = np.random.randint(20, 80) * 0.01
            data[i, :, feats_v2.index(is_col_miss)] = add_rn_missing(data[i, :, feats_v2.index(feature)], rate=r)
            list_indices = [feats_v2.index(i) for i in feats_v2 if i != is_col_miss]
            data[i, :, list_indices] = add_rn_missing(data[i, :, list_indices], rate=rate)
        else:
            if rate != 0:
                data[i] = add_rn_missing(data[i], rate=rate)
     
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std


def create_synthetic_data_v3(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (0.0001, 2 * np.pi/3, np.pi/6), # f1
        'cos': (0, np.pi, np.pi/4), # f2
        'sin-cos-plus': 0, # (0, np.pi, np.pi/10) # f3
        'sin-cos-prod': 0
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v3) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in feats_v3:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v3.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v3.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == 'sin-cos-plus':
                f1_sin = data[i, :, feats_v3.index('sin')]
                f2_cos = data[i, :, feats_v3.index('cos')]
                data[i, :, feats_v3.index(feature)] = f1_sin + f2_cos #+ np.random.rand(data.shape[1])
            elif feature == 'sin-cos-prod':
                f1 = data[i, :, feats_v3.index('sin')]
                f2 = data[i, :, feats_v3.index('cos')]
                data[i, :, feats_v3.index(feature)] = f1 * f2

            if noise and (feature != 'sin-cos-plus' or feature != 'sin-cos-prod'):
                noise_level = np.random.rand(data.shape[1]) * 0.5
                data[i, :, feats_v3.index(feature)] += noise_level

            if feature == 'sin' or feature == 'cos':
                if length_rate != 0:
                    data[i, :, feats_v3.index(feature)] = add_rn_missing(data[i, :, feats_v3.index(feature)], length=lr)
        if rate != 0:
            data[i] = add_rn_missing(data[i], rate=rate)
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std

def create_synthetic_data_v4(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/10), 
        '1+sin2': (0), 
        'cos': (-np.pi/3, np.pi/4, np.pi/10), 
        '1-cos2': (0), 
        'tan': (-np.pi/3, np.pi/3, np.pi/10), 
        '1+tan': (0)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v4) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in feats_v4:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v4.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == '1+sin2':
                f1 = data[i, :, feats_v4.index('sin')]
                data[i, :, feats_v4.index(feature)] = 1 + f1 ** 2
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v4.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == '1-cos2':
                f3 = data[i, :, feats_v4.index('cos')]
                data[i, :, feats_v4.index(feature)] = 1 - f3 ** 2
            elif feature == 'tan':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v4.index(feature)] = (np.tan(np.linspace(low, high, data.shape[1])))
            elif feature == '1+tan':
                f5 = data[i, :, feats_v4.index('tan')]
                data[i, :, feats_v4.index(feature)] = 1 + f5 ** 2
            
            if feature == 'sin' or feature == 'cos' or feature == 'tan':
                if length_rate != 0:
                    data[i, :, feats_v4.index(feature)] = add_rn_missing(data[i, :, feats_v4.index(feature)], length=lr)
            
        if rate != 0:
            data[i, :, feats_v4.index(feature)] = add_rn_missing(data[i, :, feats_v4.index(feature)], rate=rate)            
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std

def create_synthetic_data_v5(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/16), 
        '1+sin2': (0), 
        'cos': (-np.pi/3, np.pi/3, np.pi/16), 
        '1-cos2': (0), 
        'cos2-sin2': (0)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v5) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in feats_v5:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v5.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == '1+sin2':
                f1 = data[i, :, feats_v5.index('sin')]
                data[i, :, feats_v5.index(feature)] = 1 + f1 ** 2
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v5.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == '1-cos2':
                f3 = data[i, :, feats_v5.index('cos')]
                data[i, :, feats_v5.index(feature)] = 1 - f3 ** 2
            elif feature == 'cos2-sin2':
                f2 = data[i, :, feats_v5.index('1+sin2')]
                f4 = data[i, :, feats_v5.index('1-cos2')]
                data[i, :, feats_v5.index(feature)] = np.clip(f4 / f2, -30, 30)
            
            if feature == 'sin' or feature == 'cos':
                if length_rate != 0:
                    data[i, :, feats_v5.index(feature)] = add_rn_missing(data[i, :, feats_v5.index(feature)], length=lr)
            
        if rate != 0:
            data[i] = add_rn_missing(data[i], rate=rate)
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std


def create_synthetic_data_v6(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/16), 
        'cos': (-np.pi/3, np.pi/3, np.pi/16), 
        '1+sin2': (0), 
        'tan': (0)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v6) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for_sin = 0
        for_cos = 0
        for feature in feats_v6:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                linear = np.linspace(low, high, data.shape[1] + 1)
                data[i, :, feats_v6.index(feature)] = np.sin(linear[1:])
                for_sin = np.sin(linear[0])
            elif feature == '1+sin2':
                f1 = data[i, :, feats_v6.index('sin')]
                data[i, :, feats_v6.index(feature)] = 1 + f1 ** 2
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                linear = np.linspace(low, high, data.shape[1] + 1)
                data[i, :, feats_v6.index(feature)] = (np.cos(linear[1:]))
                for_cos = np.cos(linear[0])
            elif feature == 'tan':
                data[i, 0, feats_v6.index(feature)] = 1 - for_sin / for_cos
                f1 = data[i, :-1, feats_v6.index('sin')]
                f2 = data[i, :-1, feats_v6.index('cos')]
                data[i, 1:, feats_v6.index(feature)] = 1 - f1 / f2
            
            if feature == 'sin' or feature == 'cos':
                if length_rate != 0:
                    data[i, :, feats_v6.index(feature)] = add_rn_missing(data[i, :, feats_v6.index(feature)], length=lr)
            
        if rate != 0:
            data[i] = add_rn_missing(data[i], rate=rate)
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std

def create_synthetic_data_v7(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/10), 
        '1+sin2': (0), 
        'cos': (-np.pi/3, np.pi/4, np.pi/10), 
        '1-cos2': (0), 
        'tan': (-np.pi/3, np.pi/3, np.pi/10), 
        '1+tan': (0)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v7) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for feature in feats_v7:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v7.index(feature)] = np.sin(np.linspace(low, high, data.shape[1]))
            elif feature == '1+sin2':
                f1 = data[i, :, feats_v7.index('sin')]
                data[i, :, feats_v7.index(feature)] = 1 + f1 ** 2 + np.random.randn(len(data[i, :, feats_v7.index(feature)])) * 0.0001
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v7.index(feature)] = (np.cos(np.linspace(low, high, data.shape[1])))
            elif feature == '1-cos2':
                f3 = data[i, :, feats_v7.index('cos')]
                data[i, :, feats_v7.index(feature)] = 1 - f3 ** 2 + np.random.randn(len(data[i, :, feats_v7.index(feature)])) * 0.0001
            elif feature == 'tan':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                data[i, :, feats_v7.index(feature)] = (np.tan(np.linspace(low, high, data.shape[1])))
            elif feature == '1+tan':
                f5 = data[i, :, feats_v7.index('tan')]
                data[i, :, feats_v7.index(feature)] = 1 + f5 ** 2 + np.random.randn(len(data[i, :, feats_v7.index(feature)])) * 0.0001
            
            if feature == 'sin' or feature == 'cos' or feature == 'tan':
                if length_rate != 0:
                    data[i, :, feats_v4.index(feature)] = add_rn_missing(data[i, :, feats_v4.index(feature)], length=lr)
        if rate != 0:
            data[i, :, feats_v7.index(feature)] = add_rn_missing(data[i, :, feats_v7.index(feature)], rate=rate)
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std

def create_synthetic_data_v8(n_steps, num_seasons, seed=10, rate=0.05, length_rate=0.02, noise=False):
    # num_seasons = 32
    np.random.seed(seed)
    
    synth_features = {
        'sin': (-np.pi/3, np.pi/3, np.pi/16), 
        'cos': (-np.pi/3, np.pi/3, np.pi/16), 
        'tan': (0)
    }
    num_steps = n_steps # config['n_steps']
    num_features = len(feats_v8) # config['n_features']
    data = np.zeros((num_seasons, num_steps, num_features))
    # value_range = [(0.1, 0.4, 0.7, 0.99), (11.0, 17.5, 40.5, 61.2), (100.1, 160.2, 500, 1000)]
    
    for i in range(data.shape[0]):
        for_sin = 0
        for_cos = 0
        for feature in feats_v8:
            lr = int(np.floor(n_steps * (np.random.rand() * length_rate)))
            args = synth_features[feature]
            if feature == 'sin':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                linear = np.linspace(low, high, data.shape[1] + 1)
                data[i, :, feats_v8.index(feature)] = np.sin(linear[1:])
                for_sin = np.sin(linear[0])
            elif feature == 'cos':
                low = np.random.uniform(args[0], args[0] + args[2])
                high = np.random.uniform(args[1], args[1] + args[2])
                linear = np.linspace(low, high, data.shape[1] + 1)
                data[i, :, feats_v8.index(feature)] = (np.cos(linear[1:]))
                for_cos = np.cos(linear[0])
            elif feature == 'tan':
                data[i, 0, feats_v8.index(feature)] = 1 - for_sin / for_cos
                f1 = data[i, :-1, feats_v8.index('sin')]
                f2 = data[i, :-1, feats_v8.index('cos')]
                data[i, 1:, feats_v8.index(feature)] = 1 - f1 / f2 + data[i, 0:-1, feats_v8.index(feature)]
            
            if feature == 'sin' or feature == 'cos':
                if length_rate != 0:
                    data[i, :, feats_v8.index(feature)] = add_rn_missing(data[i, :, feats_v8.index(feature)], length=lr)
            
        if rate != 0:
            data[i] = add_rn_missing(data[i], rate=rate)
    
    data_rows = data.reshape((-1, num_features))
    mean = np.nanmean(data_rows, axis=0)
    std = np.nanstd(data_rows, axis=0)
    data = data.reshape((num_seasons, num_steps, num_features))
    return data, mean, std