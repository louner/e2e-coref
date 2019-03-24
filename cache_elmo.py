from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import json
import sys

import tensorflow_hub as hub
from bert import tokenization
import tensorflow as tf
#BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"

from pdb import set_trace

def build_elmo():
  token_ph = tf.placeholder(tf.string, [None, None])
  len_ph = tf.placeholder(tf.int32, [None])
  elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
  lm_embeddings = elmo_module(
      inputs={"tokens": token_ph, "sequence_len": len_ph},
      signature="tokens", as_dict=True)
  word_emb = lm_embeddings["word_emb"]
  lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                     lm_embeddings["lstm_outputs1"],
                     lm_embeddings["lstm_outputs2"]], -1)
  return token_ph, len_ph, lm_emb

def build_bert():
    input_ids = tf.placeholder(tf.int32, [None, None])
    input_mask = tf.placeholder(tf.int32, [None, None])
    segment_ids = tf.placeholder(tf.int32, [None, None])

    bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)

    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)

    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

    return input_ids, input_mask, segment_ids, bert_outputs["sequence_output"]

def cache_dataset(data_path, session, input_ids, input_mask, segment_ids, sequence_output, out_file):
  with open(data_path) as in_file:
    for doc_num, line in enumerate(in_file.readlines()):
      example = json.loads(line)

      #set_trace()
      tf_lm_emb = session.run(sequence_output, feed_dict={
        input_ids: example["input_ids"],
        input_mask: example["input_mask"],
        segment_ids: example["segment_ids"]
      })

      file_key = example["doc_key"].replace("/", ":")
      group = out_file.create_group(file_key)
      for i, (e, mask) in enumerate(zip(tf_lm_emb, example['input_mask'])):
        mask = np.array(mask).astype(bool)
        e = e[mask]
        group[str(i)] = e
      if doc_num % 10 == 0:
        print("Cached {} documents in {}".format(doc_num + 1, data_path))

if __name__ == "__main__":
  #token_ph, len_ph, lm_emb = build_elmo()
  input_ids, input_mask, segment_ids, sequence_output = build_bert()
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    with h5py.File("elmo_cache.hdf5", "w") as out_file:
      for json_filename in sys.argv[1:]:
        #cache_dataset(json_filename, session, token_ph, len_ph, lm_emb, out_file)
        cache_dataset(json_filename, session, input_ids, input_mask, segment_ids, sequence_output, out_file)
