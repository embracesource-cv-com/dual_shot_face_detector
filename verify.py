# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: verify.py
@author: danna.li
@time: 2019-06-04 09:57
@description: 
"""
import keras.layers as KL
from keras import Model
from dual_conf import current_config as conf
from net.dual_shot import test_net
from prepare_data.generator import gen_test
import os
from prepare_data.model_target import apply_regress


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
weight_path = os.path.join(conf.output_dir, 'weights.h5')
print('loading trained model from:',weight_path)

net_in = KL.Input([640, 640, 3], name='image_array')
ss_cls, ss_regr = test_net(net_in)
model = Model(inputs=[net_in], outputs=[ss_cls, ss_regr])
model.summary()
model.load_weights(weight_path, by_name=True)



gen = gen_test(conf.batch_size, 'train')
x_val = next(gen)[0]
print(x_val.shape)
ss_cls,ss_regr= model.predict(x_val)
print(ss_cls.shape,ss_regr.shape)





