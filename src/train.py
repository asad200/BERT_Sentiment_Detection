import os
import prep
from model import create_model
import config

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from bert.tokenization.bert_tokenization import FullTokenizer



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
max_seq_len = 192



df = pd.read_csv('input/IMDB_Dataset.csv')
train, test = train_test_split(df, test_size=0.2, random_state=seed)

classes = train.sentiment.unique().tolist()

# get the tokenizer from bert model
tokenizer = FullTokenizer(vocab_file=os.path.join(config.bert_ckpt_dir, 'vocab.txt'))

data = prep.IntentRecognition(train, test, tokenizer, classes, max_seq_len=128)

model = create_model(max_seq_len, classes, config.bert_config_file, config.bert_ckpt_file)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config.checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

history = model.fit(
    x=data.x_train,
    y=data.y_train,
    validation_split=0.1,
    batch_size=16,
    shuffle=True,
    epochs=5,
    callbacks=[model_checkpoint_callback]
    )

_, test_acc = model.evaluate(data.x_test, data.y_test)
print(test_acc)

predictions = model.predict(data.x_test) >= 0.5

print(classification_report(data.y_test, predictions, target_names=classes))

# Save the entire model as a SavedModel.
model.save('saved_model/my_model') 