# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/29 
@file: train_dual.py
@description:
"""

from prepare_data.generator import gen_data
from dual_conf import current_config as conf
from vgg_ssd import train_net
import keras
from keras import Model, Input
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index

log = TensorBoard(log_dir=conf.output_dir, )
lr_decay = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.95)
ckpt_saver = ModelCheckpoint(monitor='loss', filepath=os.path.join(conf.output_dir, 'weights.h5'), verbose=1,
                             save_best_only=True, save_weights_only=True)
callback = [log, lr_decay, ckpt_saver]

gen = gen_data(conf.batch_size, 'train')
print([i.shape for i in next(gen)[0]])
num_anchor = conf.num_train_anchor
x_in = Input([conf.net_in_size, conf.net_in_size, 3], name='image_array')
y_e_reg = Input((num_anchor, 5), name='e_reg')
y_e_ind = Input((num_anchor, 2), name='e_train_ind')
y_o_reg = Input((num_anchor, 5), name='o_reg')
y_o_ind = Input((num_anchor, 2), name='o_train_ind')
fs_cls, fs_regr, ss_cls, ss_regr, pal = train_net(x_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind)
model = Model(inputs=[x_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind], outputs=[fs_cls, fs_regr, ss_cls, ss_regr, pal])

loss_name = 'PAL'
layer = model.get_layer(loss_name)
loss = layer.output
model.add_loss(loss)
model.summary()

if conf.continue_training:
    print('loading trained weights from..', conf.weights_to_transfer)
    model.load_weights(conf.weights_to_transfer, by_name=True)

model.compile(optimizer=Adam(lr=conf.lr, decay=conf.weight_decay), loss=None)
model.fit_generator(generator=gen,
                    validation_steps=2,
                    steps_per_epoch=conf.steps_per_epoch,
                    epochs=conf.epochs,
                    callbacks=callback)
