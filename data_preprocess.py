import tensorflow as tf
import numpy as np
import h5py


datapath = '...' # data file path
def data_load(index):
    directory = datapath + '\\d'
    x_data, y_data = [], []
    x_data = np.array(x_data)

    for fileIndex in range(index):
        filename = directory + str(fileIndex + 1) + '.mat'
        feature = h5py.File(filename, mode='r')
        a = list(feature.keys())
        x_sub = feature[a[0]]
        x_sub = np.array(x_sub)
        if x_data.size == 0:
            x_data = x_sub
        else:
            x_data = np.concatenate((x_data, x_sub), axis=2)

    feature = h5py.File(datapath + '\\y_stim.mat', mode='r')
    a = list(feature.keys())
    y_data = feature[a[0]]
    y_data = np.array(y_data)

    return x_data, y_data

# ——————————————————k折交叉验证划分——————————————————————
def get_k_fold_data(k, i, X, y):
    assert k > 1
    X, y = easy_shuffle(X, y)
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = tf.concat([X_train, X_part], 0)
            y_train = tf.concat([y_train, y_part], 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    return X_train, y_train, X_valid, y_valid

# ——————————————————数据降维—————————————————————
def flatten_test(input_arr, output_arr=None):
    if output_arr is None:
        output_arr = []
    for ele in input_arr:
        if not isinstance(ele, str) and isinstance(ele, Iterable):
            flatten_test(ele, output_arr)  # tail-recursion
        else:
            output_arr.append(ele)  # produce the result
    return output_arr

# ——————————————————简易打乱数据顺序—————————————————————
def easy_shuffle(X, y):
    index = [i for i in range(X.shape[0])]
    np.random.shuffle(index)
    print('shuffle index: ', index)
    X = X[index]
    y = y[index]
    return X, y

# ——————————————————将数据 按受试者标签id 划分—————————————————————
def split_data(X_data, y_index, y_label, subject_id, isShuffle=False):
    # split subject, acquire test-subject data to test model
    # y_test_idx = []
    # for i in subject_id:
    #     y_test_idx.append(np.where(y_index == i))  # get test-subject index in y_data
    #
    # y_test_idx = flatten_test(y_test_idx)
    # y_test_idx = np.array(y_test_idx)

    print('needed split_idx: ', y_index)
    print('subjects-id: ', subject_id)
    train_part_x, train_part_y, test_part_x, test_part_y = [], [], [], []
    i_idx = 0
    for i in y_index:
        if i in subject_id:
            test_part_x.append(X_data[i_idx])
            test_part_y.append(y_label[i_idx])
            # print('choose sub index: ',  i_idx)
        else:
            train_part_x.append(X_data[i_idx])
            train_part_y.append(y_label[i_idx])
        i_idx += 1

    train_part_x = np.array(train_part_x)
    train_part_y = np.array(train_part_y)
    test_part_x = np.array(test_part_x)
    test_part_y = np.array(test_part_y)
    if isShuffle:
        train_part_x, train_part_y = easy_shuffle(train_part_x, train_part_y)
        test_part_x, test_part_y = easy_shuffle(test_part_x, test_part_y)
    return train_part_x, train_part_y, test_part_x, test_part_y
