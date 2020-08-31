import numpy as np
from tqdm import tqdm
from bert.tokenization.bert_tokenization import FullTokenizer

class IntentRecognition:
  DATA_COL = 'review'
  LABEL_COL = 'sentiment'

  def __init__(self, train, test, tokenizer:FullTokenizer, max_seq_len=192):
    self.tokenizer = tokenizer
    self. max_seq_len = 0

    ((self.x_train, self.y_train), (self.x_test, self.y_test)) = map(self._prepare, [train, test])

    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.x_train, self.x_test = map(self._pad, [self.x_train, self.x_test])

  def _prepare(self, df):
    x, y = [], []

    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentRecognition.DATA_COL], row[IntentRecognition.LABEL_COL]

      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]

      # conver tokens to token ids (to numeric values)
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

      # get the max_seq_len
      self.max_seq_len = max(self.max_seq_len, len(token_ids))

      x.append(token_ids)

    return np.array(x), np.array(df[IntentRecognition.LABEL_COL])

  def _pad(self, ids):
    x = []
    for id in ids:
      cut_point = min(len(id), self.max_seq_len)
      id = id[:cut_point]
      id = id + [0] * (self.max_seq_len - len(id))
      x.append(id)

    return np.array(x)