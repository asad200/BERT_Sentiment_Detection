import os
import numpy as np
import tensorflow as tf



bert_model_name = 'uncased_L-12_H-768_A-12'
bert_ckpt_dir = os.path.join('input/model/', bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, 'bert_model.ckpt')
bert_config_file = os.path.join(bert_ckpt_dir, 'bert_config.json')

checkpoint_filepath = 'output/weights/checkpoint'