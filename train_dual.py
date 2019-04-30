# -*- coding:utf-8 _*-
"""
@author: danna.li
@date: 2019/4/29 
@file: train_dual.py
@description:
"""

from prepare_data.generator import gen_data
from dual_conf import current_config as conf
from vgg_ssd import whole_net
import keras
from keras import Model, Input
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu_index

log = TensorBoard(log_dir=conf.output_dir)
gen = gen_data(conf.batch_size)
print([i.shape for i in next(gen)[0]])
num_anchor = conf.num_train_anchor
x_in = Input([conf.net_in_size, conf.net_in_size, 3], name='image_array')
y_e_reg = Input((num_anchor, 5), name='e_reg')
y_e_ind = Input((num_anchor, 2), name='e_train_ind')
y_o_reg = Input((num_anchor, 5), name='o_reg')
y_o_ind = Input((num_anchor, 2), name='o_train_reg')
pal = whole_net(x_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind)
model = Model(inputs=[x_in, y_e_reg, y_e_ind, y_o_reg, y_o_ind], outputs=pal)

name = 'PAL'
layer = model.get_layer(name)
loss = layer.output
model.add_loss(loss)

model.summary()
model.compile(optimizer=Adam(lr=conf.lr, decay=conf.weight_decay), loss=None)
model.fit_generator(generator=gen,
                    validation_steps=2,
                    steps_per_epoch=conf.steps_per_epoch,
                    epochs=conf.epochs,
                    callbacks=[log])
