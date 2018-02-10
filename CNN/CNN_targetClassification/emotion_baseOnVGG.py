#coding=utf-8
# @Author: yangenneng
# @Time: 2018-02-10 10:56
# @Abstract：基于VGG进行人脸表情识别

import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
import os

#重新定义VGG
def vgg16(input,output):
    activation_function='relu' #激活函数

    # scope:define this layer scope (optional).
    # activation: 激活函数
    # trainable: bool量，是否可以被训练 trainable=False表示不会更新本层的参数
    x = tflearn.conv_2d(input, 64, 3, activation=activation_function, scope='conv1_1', trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation=activation_function, scope='conv1_2', trainable=False)
    #最大池化操作
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation=activation_function, scope='conv2_1', trainable=False)
    x = tflearn.conv_2d(x, 128, 3, activation=activation_function, scope='conv2_2', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation=activation_function, scope='conv3_1', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation=activation_function, scope='conv3_2', trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation=activation_function, scope='conv3_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv4_1', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv4_2', trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv4_3', trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation=activation_function, scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    # 全连接层
    x = tflearn.fully_connected(x, 4096, activation=activation_function, scope='fc6')
    # dropout操作
    x = tflearn.dropout(x, 0.5, name='dropout1')

    #此处原来是4096 现在改为2048，减小参数
    x = tflearn.fully_connected(x, 2048, activation=activation_function, scope='fc7', restore=False)
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, output, activation='softmax', scope='fc8', restore=False)

    return x

#模型路径
model_path="."
#数据列表
files_list = "./train_fvgg_emo.txt"

#(X, Y): with X the images array and Y the labels array.
X,Y=image_preloader(files_list,image_shape=(224,224),mode='file',
                    categorical_labels=True,normalize=False,
                    files_extension=['.jpg', '.png'],filter_channel=True
                    )

output=7 #输出分类

# VGG 图像预处理
img_pre=ImagePreprocessing()
# 确定数据是规范的
img_pre.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],per_channel=True)

# VGG Network building
net=tflearn.input_data(shape=[None, 224, 224, 3],name='input',data_preprocessing=img_pre)

softmax=vgg16(net,output)

regression=tflearn.regression(softmax,optimizer='adam',
                              learning_rate=0.01,restore=False
                              )
#checkpoint_path：要评估的特定检查点的路径。如果为 None，则使用 model_dir 中的最新检查点。
model=tflearn.DNN(regression,checkpoint_path='vgg-finetuning',
                  max_checkpoints=3,tensorboard_verbose=2,
                  tensorboard_dir="./logs"
                  )

model_file=os.path.join(model_path,'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

model.load(model_file,weights_only=True)


# Start finetuning
model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_epoch=False,
          snapshot_step=200, run_id='vgg-finetuning')

model.save('vgg_finetune_emo.tfmodel')


