import numpy as np
import pandas as pd
import json
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class MySet(Dataset):
    def __init__(self, filename='./json/json'):
        super(MySet, self).__init__()
        self.content = open(filename).readlines()

        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        # print(f"rec type: {type(rec)}")
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def parse_delta(masks, n_features, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(n_features))
        else:
            deltas.append(np.ones(n_features) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, n_features, dir_):
    deltas = parse_delta(masks, n_features, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_id(fs, x, n_features):
    evals = x
    # evals = (evals - mean) / std

    shp = evals.shape

    evals = evals.reshape(-1)
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 20)
    

    values = evals.copy()
    values[indices] = np.nan

    masks = ~np.isnan(values)

    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)
    # label = y.tolist() #out.loc[int(id_)]
    # print(f'rec y: {list(y)}')
    # rec = {'label': label}
    rec = {}
    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, n_features, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], n_features, dir_='backward')
    # for key in rec.keys():
    #     print(f"{key}: {type(rec[key])}")# and {rec[key].shape}")
    rec = json.dumps(rec)

    fs.write(rec + '\n')

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    # print(f"forward: {forward}")
    backward = list(map(lambda x: x['backward'], recs))
    # print(f"backward: {backward.__next__()['masks']}")
    def to_tensor_dict(recs):
        # print(f"recs: {list(recs['masks'])}")
        values = list(map(lambda r: r['values'], recs))
        # print(f"r[values]: {value_list}")
        values = torch.FloatTensor(values)
        # print(f"values: {values}")
        masks = list(map(lambda r: r['masks'], recs))
        
        masks = torch.FloatTensor(masks)
        # print(f"masks: {masks.shape}")
        deltas = list(map(lambda r: r['deltas'], recs))
        deltas = torch.FloatTensor(deltas)
        # print('deltas: ', deltas.shape)

        evals = list(map(lambda r: r['evals'], recs))
        evals = torch.FloatTensor(evals)
        eval_masks = list(map(lambda r: r['eval_masks'], recs))
        eval_masks = torch.FloatTensor(eval_masks)
        forwards = list(map(lambda r: r['forwards'], recs))
        forwards = torch.FloatTensor(forwards)

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    # labels = list(map(lambda x: x['label'], recs))
    # print('loader label: ', labels)
    # ret_dict['labels'] = torch.FloatTensor(labels)
    # print('loader rec label: ', ret_dict['labels'].shape)
    is_trains = list(map(lambda x: x['is_train'], recs))
    ret_dict['is_train'] = torch.FloatTensor(is_trains)

    return ret_dict

def get_loader(batch_size = 64, shuffle = True, filename='./json/json'):
    data_set = MySet(filename)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 0, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter

def brits_train(model, n_epochs, batch_size, model_path, data_file='./json/json'):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    data_iter = get_loader(batch_size=batch_size, filename=f"{data_file}")
    pre_mse = 9999999
    count_diverge = 0
    for epoch in range(n_epochs):
        model.train()
        if count_diverge > 3:
            break
        with tqdm(data_iter, unit='batch') as tepoch:
            run_loss = 0.0
            tepoch.set_description(f"Epoch {epoch+1}/{n_epochs} [T]")
            for idx, data in enumerate(data_iter):
                data = to_var(data)
                ret = model.run_on_batch(data, optimizer, epoch)

                run_loss += ret['loss'].item()
                tepoch.set_postfix(train_loss=(run_loss / (idx + 1.0)))
                tepoch.update(1)
                # print('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

            mse = brits_evaluate(model, data_iter)
            if pre_mse < mse:
                count_diverge += 1
            else:
                count_diverge = 0
            tepoch.set_postfix(train_loss=(run_loss / (idx + 1.0)), val_loss=mse)
            tepoch.update(1)
        if (epoch + 1) % 100 == 0 and count_diverge == 0:
            torch.save(model.state_dict(), model_path)
    end = time.time()
    print(f"time taken for training: {end-start}s")
    return model

def brits_evaluate(model, val_iter):
    model.eval()
    evals = []
    imputations = []

    save_impute = []
    for idx, data in enumerate(val_iter):
        data = to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    mse = ((evals - imputations) ** 2).mean()
    # print('MSE: ', mse)
    # save_impute = np.concatenate(save_impute, axis=0)
    # if not os.path.isdir('./result_brits/'):
    #     os.makedirs('./result_brits/')
    # np.save('./result/data_brits', save_impute)
    # np.save('./result/label', save_label)
    return mse

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def prepare_brits_input(filename, X):
    # X = X[:-2]
    # Y = Y[:-2]
    fs = open(filename, "w")
    for i in range(X.shape[0]):
        parse_id(fs, X[i], X[i].shape[1])

    fs.close()

def parse_id_test(fs, x, n_features, target, obs_mask, target_mask):

    # prepare the model for both directions
    values = x
    evals = target
    masks = obs_mask
    eval_masks = target_mask
    rec = {}
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, n_features, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], n_features, dir_='backward')

    rec = json.dumps(rec)

    fs.write(rec + '\n')


def test_evaluate(model, filename, X, target, obs_mask, target_mask):
    # X = X[:-2]
    # Y = Y[:-2]
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(obs_mask, torch.Tensor):
        obs_mask = obs_mask.cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()

    fs = open(filename, 'w')
    for i in range(X.shape[0]):
        parse_id_test(fs, X[i], X[i].shape[1], target[i], obs_mask[i], target_mask[i])
    fs.close()
    test_loader = get_loader(batch_size = X.shape[0], shuffle = False, filename=filename)
    imputation = None
    for idx, data in enumerate(test_loader):
        data = to_var(data)
        ret = model.run_on_batch(data, None)
        imputation = ret['imputations']
    return imputation