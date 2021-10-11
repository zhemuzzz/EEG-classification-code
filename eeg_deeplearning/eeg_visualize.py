import tensorflow as tf
import numpy as np
import h5py
import os
import cv2
from scipy.io import savemat
from matplotlib import pyplot as plt



def get_saliency_map(model, image, class_idx):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    # print('image.shape: ', image.shape)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[:, class_idx]

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, image)

    # take maximum across channels
    gradient = tf.reduce_max(gradient, axis=-1)

    # convert to numpy
    gradient = gradient.numpy()

    # normaliz between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

    return smap
  
  
  
  def test_visualize(params, X_test, y_test):
    trial_num = X_test.shape[0]
    X_test = X_test.reshape(trial_num, params['chans'], params['samples'], params['kernels'])
    test = X_test[0:trial_num, :, :, :]
    y_res = y_test[0:trial_num]

    # test = np.mean(test, axis=0)
    test = np.reshape(test, (trial_num, 56, 385, 1))
    print('test.shape', test.shape)
    print('y_res', y_res)

    # visualize
    model = tf.keras.models.load_model(params['model_weights_path'])
    # layer_outputs = []
    # for i, layer in enumerate(model.layers):
    #     if i in [1, 3, 9, 11, 13]:
    #         print('layer.name: ', layer.name)
    #         layer_outputs.append(layer.output)
    # activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # activation_model.summary()
    # activations = activation_model.predict(test)

    # 画出原图
    origin = np.squeeze(test)
    origin = origin / np.max(origin)
    # plt.figure(figsize=(10, 10))
    # plt.plot(1, 1, 1)
    # # plt.plot(origin.swapaxes(1, 0))
    # plt.imshow(origin, cmap='hot', vmin=-1, vmax=1)
    # print('origin map: ', origin)

    # -------------------------------------layer output------------------
    # xtick = np.arange(0, 385 / 256, 1 / 256)
    # # conv2d 第一个卷积的特征图
    # featuremaps = np.transpose(activations[0], (3, 0, 2, 1)).squeeze()
    # print('featuremaps1.shape: ', featuremaps.shape)
    # plt.figure(figsize=(40, 20))
    # # for i, featuremap in enumerate(featuremaps):
    # #     plt.subplot(5, 10, i + 1)
    # #     plt.plot(xtick, featuremap)
    # featuremap = np.mean(featuremaps, axis=0).squeeze()
    # plt.plot(xtick, featuremap)
    #
    # # conv2d 第二个卷积的特征图
    # featuremaps = np.transpose(activations[1], (3, 0, 2, 1)).squeeze()
    # print('featuremaps2.shape: ', featuremaps.shape)
    # plt.figure(figsize=(40, 20))
    # # for i, featuremap in enumerate(featuremaps):
    # #     plt.subplot(10, 10, i + 1)
    # #     plt.plot(xtick, featuremap)
    # featuremap = np.mean(featuremaps, axis=0).squeeze()
    # plt.plot(xtick, featuremap)
    #
    # # lstm1 第1个lstm层特征图
    # featuremaps = np.squeeze(activations[2])
    # print('featuremaps3.shape: ', featuremaps.shape)
    # plt.figure(figsize=(40, 20))
    # plt.plot(featuremaps)
    #
    # # lstm2 第2个lstm层特征图
    # featuremaps = np.squeeze(activations[3])
    # print('featuremaps4.shape: ', featuremaps.shape)
    # plt.figure(figsize=(20, 10))
    # plt.plot(featuremaps)
    #
    # # 预测输出
    # preds = activations[4]
    # print('preds.shape: ', preds.shape)
    # print('preds : ', preds)
    #
    # plt.show()
    #
    # # ---------------------------------saliency map-----------------------
    # 通过网络层的名字找到layer_idx
    linear_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    class_idx = y_res.argmax(axis=-1)
    print('class_idx: ', class_idx)
    grads = []
    for idx in range(trial_num):
        grads.append(get_saliency_map(linear_model, np.reshape(test[idx], (1, 56, 385, 1)), class_idx[idx]))
    # grads.append(get_saliency_map(linear_model, np.reshape(test, (1, 56, 385, 1)), class_idx[1]))  # 此处为1次试次trial的简化，可能经过mean处理
    # heatmap = np.squeeze(grads)
    # print('grads.shape: ', heatmap.shape)
    #
    # plt.figure(figsize=(10, 10))
    # plt.su    # # plt.plot(np.squeeze(grads).swapaxes(1, 0))bplot(1, 1, 1)
    # plt.imshow(heatmap, cmap='hot', vmin=-1, vmax=1)
    # print('grad map: ', heatmap)

    # # -----------------------画重叠图-------------------------------
    # super_img = cv2.addWeighted(origin, 0.5, heatmap, 0.5, 0)
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # # plt.plot(super_img.swapaxes(1, 0))
    # plt.imshow(super_img, cmap='hot', vmin=-1, vmax=1)
    # plt.show()

    # ---------------------保存数据 转换成mat格式 -----------------------
    # conv1 = np.transpose(activations[0], (3, 0, 1, 2)).squeeze()
    # conv2 = np.transpose(activations[1], (3, 0, 1, 2)).squeeze()
    # lstm1 = np.squeeze(activations[2])
    # lstm2 = np.squeeze(activations[3])
    # print('conv1.shape', conv1.shape)
    # print('conv2.shape', conv2.shape)
    # print('lstm1.shape', lstm1.shape)
    # print('lstm2.shape', lstm2.shape)

    grads = np.squeeze(grads)
    print('grads.shape', grads.shape)

    # origin = np.mean(origin, axis=0)
    # grads = np.mean(grads, axis=0)
    
    # 根据main函数中 测试可视化数据来源 给数据起名字 对应上即可
    file_name = 'sub-add-5.mat'
    savemat(file_name, {'origin': origin, 'grads': grads})

  
  
  
