
import tensorflow as tf
import numpy as np
import h5py
import os
from EEG_models import EEGNet, ShallowConvNet, square, log, DeepConvNet, cnnlstm
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt

from data_preprocess import data_load, get_k_fold_data, flatten_test, easy_shuffle, split_data


def gpu_start():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # cpu
    # todo:解决tensorflow后端显存占用的bug
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in range(4):
        print('gpu', i)
        tf.config.experimental.set_memory_growth(physical_devices[i], True)


def train_eeg_model(params, train_dataset, valid_dataset, fold_index):
    chans, samples = params['chans'], params['samples']
    # multi gpu
    strategy = params['strategy']
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    with strategy.scope():
        # use EEGNet
        # model = EEGNet(nb_classes=3, Chans=chans, Samples=samples,
        #                dropoutRate=0.2, kernLength=64, F1=8, D=2, F2=16,
        #                dropoutType='Dropout')
        # model = ShallowConvNet(nb_classes=3, Chans=chans, Samples=samples, weight_decay=1)
        # print('Use the ShallowConvNet model............')

        model = cnnlstm(nb_classes=3, Chans=chans, Samples=samples, dropoutRate=0.5, weight_decay=0.1)
        print('Use the CNN-LSTM model..................')

        # model = DeepConvNet(nb_classes=3, Chans=chans, Samples=samples, dropoutRate=0.5, weight_decay=1)

        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=params['lr']),
                      metrics=['accuracy'])

    models_path = os.path.join(params['output_dir'], 'saved_models')
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    save_model_weights_path = models_path + '/model-weights-' + str(fold_index) + '.hdf5'

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=save_model_weights_path, verbose=1, monitor='val_loss',
                                   mode='auto', save_best_only=True)
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0: 1, 1: 1, 2: 1}

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model_history = model.fit(train_dataset, epochs=params['num_epochs'],
                              verbose=2, validation_data=valid_dataset,
                              callbacks=[checkpointer, reduce_lr, early_stop], class_weight=class_weights)

    plot_history(model_history, params['output_dir'], fold_index)
    save_history(model_history, params['output_dir'], fold_index)
    model.summary()

    model_structure_json = model.to_json()
    save_models_structure_path = os.path.join(models_path, 'model_architecture' + str(fold_index) + '.json')
    open(save_models_structure_path, 'w').write(model_structure_json)
    params['model_structure_path'] = save_models_structure_path
    params['model_weights_path'] = save_model_weights_path


def test_main(params, test_x, test_y, fold_index):
    probs, real, acc = test_model(params, X_test=test_x, Y_test=test_y, fold_index=fold_index)
    # todo: plot confusion matrix, plot ROC curve

def test_model(params, X_test, Y_test, fold_index):
    X_test = X_test.reshape(X_test.shape[0], params['chans'], params['samples'], params['kernels'])
    model = tf.keras.models.load_model(params['model_weights_path'], custom_objects={'square': square, 'log': log})
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    real = Y_test.argmax(axis=-1)
    acc = np.mean(preds == real)
    params['acc'].append(acc.item())
    print("Classification accuracy: %f " % acc)
    # todo: record test result
    if params['mode_'] == 'k_fold':
        filename = 'k-fold-result-' + str(fold_index) + '.txt'
    else:
        filename = 'LOOS-result-' + str(fold_index) + '.txt'
    with open(os.path.join(params['output_dir'], filename), 'w') as fp:
        fp.write('trial_num\tHC\tADD\tADHD-C\tpred_label\treal_label\n')
        for i in range(len(preds)):
            fp.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                i, probs[i][0], probs[i][1], probs[i][2], preds[i], real[i]))
    return probs, real, acc


def train_main(params, X_data, y_data):
    # ---------------------------- k fold validation part------------------------
    X_data, y_data = easy_shuffle(X_data, y_data)
    X_test, y_test = X_data[0:6000, :], y_data[0:6000, 1:4]
    print('y_test:  ', y_test)
    print('y_test.shape', y_test.shape)
    kpart_x, kpart_y = X_data[6001:, :], y_data[6001:, 1:4]

    params['mode_'] = 'k_fold'
    k = params['k']
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, kpart_x, kpart_y)
        print('the fold times: ', i + 1)
        print('X_train shape:', X_train.shape)
        print('X_valid shape:', X_valid.shape)
        print('y_train shape:', y_train.shape)
        print('y_valid shape:', y_valid.shape)
        print('batch_size: ', params['batch_size'])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0]).batch(params['batch_size'])

        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        valid_dataset = valid_dataset.shuffle(buffer_size=X_valid.shape[0]).batch(params['batch_size'])

        train_eeg_model(params=params, train_dataset=train_dataset, valid_dataset=valid_dataset, fold_index=i + 1)
        print('ready for testing....')
        test_main(params=params, test_x=X_test, test_y=y_test, fold_index=i)
    print('cross subjects test was finished.............: ')
    print('cross subjects all acc list: ', params['acc'])
    print('cross subjects mean acc: ', np.mean(params['acc']))

    # ------------------------------ leave one subject out----------------
    # params['mode_'] = 'leaveOneOut'
    # y_label = y_data[:, 1:4]
    # y_index = y_data[:, 0]
    # y_index = np.array(y_index, dtype=int)
    # # todo: change the raw y_index
    # for i in range(y_index.shape[0]):
    #     if params['HC_trials'] < i + 1 <= params['HC_trials'] + params['ADD_trials']:
    #         y_index[i] += params['HC_num']
    #     if i + 1 > params['HC_trials'] + params['ADD_trials']:
    #         y_index[i] += params['HC_num'] + params['ADD_num']
    #
    # for i in range(params['subject_num']):
    #     subject_id = np.array([i + 1])
    #     train_part_x, train_part_y, test_x, test_y = split_data(X_data, y_index, y_label, subject_id, False)
    #     print('the test subject-id: ', subject_id)
    #     print('test_x shape: ', test_x.shape)
    #     print('test_y shape: ', test_y.shape)
    #
    #     valid_subject = 20
    #     print('now choose ' + str(valid_subject) + ' subjects randomly to validation subject....')
    #     subjects = np.arange(1, params['subject_num'] + 1)
    #     subjects = np.delete(subjects, np.where(subjects == i + 1))
    #     delete_index = np.where(y_index == i + 1)
    #     train_part_index = np.delete(y_index, delete_index, 0)
    #
    #     print('train_part_index.shape: ', train_part_index.shape)
    #     print('train_part_y.shape: ', train_part_y.shape)
    #     subjects_id = np.random.choice(subjects, valid_subject, replace=False)
    #     print('chooses ' + str(valid_subject) + ' subject for validation , id finished: ', subjects_id)
    #     train_x, train_y, valid_x, valid_y = split_data(train_part_x, train_part_index, train_part_y, subjects_id, True)
    #     print('train_x shape: ', train_x.shape)
    #     print('valid_x shape: ', valid_x.shape)
    #     print('train_y.shape: ', train_y.shape)
    #
    #     train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    #     train_dataset = train_dataset.shuffle(buffer_size=20000).batch(params['batch_size'])
    #
    #     valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
    #     valid_dataset = valid_dataset.shuffle(buffer_size=2000).batch(params['batch_size'])
    #     print('ready for training.... params: ', params)
    #     train_eeg_model(params=params, train_dataset=train_dataset, valid_dataset=valid_dataset, fold_index=i + 1)
    # print('ready for testing.... params: ', params)
    #     test_main(params=params, test_x=test_x, test_y=test_y, fold_index=i + 1)
    # print('144 subjects test was finished.............: ')
    # print('144 subjects all acc list: ', params['acc'])
    # print('144 subjects mean acc: ', np.mean(params['acc']))
   
    

if __name__ == '__main__':
    gpu_start()
    # --------------------------------------------- param dict------------------------
    params = {
        'subject_num': 144, 'HC_num': 44, 'ADD_num': 52, 'ADHD_num': 48,
        'HC_trials': 10129, 'ADD_trials': 13031, 'ADHD_trials': 10742,
        'output_dir': r'E:\wangcheng\Experiment-01-out',
        # 'strategy': tf.distribute.MultiWorkerMirroredStrategy(),
        # tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        'num_epochs': 300, 'batch_size': 256,
        'k': 5, 'lr': 1e-3,
        'kernels': 1, 'chans': 56, 'samples': 385,
        'mode_': 'k-fold', 'acc': [],
        'model_weights_path': '', 'model_structure_path': ''
    }
    # ----------------------------------------------data load------------------------
    X_data, y_data = data_load(7)
    X_data = np.swapaxes(X_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    print('x_data.shape: ', X_data.shape)
    print('y_data.shape: ', y_data.shape)

    # todo: model training
    # train_main(params, X_data, y_data)

    # ---------------------------------------------test visualize---------------------------------------------------------

    # print('y_test subject index : ', y_data[11000:19000, 0])
    X_test, y_test, y_idx = X_data[11150:11400, :], y_data[11150:11400, 1:4], y_data[11150:11400, 0]

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    params['model_weights_path'] = r'C:\Users\siat-sj1\Desktop\SCIresult\model_result\13713-9823-cnnlstm\model-weights-1.hdf5'
    
    test_visualize(params, X_test, y_test)
    
