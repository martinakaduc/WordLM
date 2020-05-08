import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from gensim import models
import numpy as np
import copy

def text_cleaner(text):
    # lower case text
    text = text.lower()
    text = re.sub(r"'s\b","",text)
    # remove punctuations
    # INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    text = re.sub("[^a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]", " ", text)
    return text

def encode_sequence(mapping, text, seq_length, padding=False):
    encoded_seq = []
    for char in text:
        if char in mapping:
            # print('RUNING')
            encoded_seq.append(copy.deepcopy(mapping[char]))
        else:
            encoded_seq.append([0]*300)

    if padding:
        # encoded_seq = pad_sequences([encoded_seq], maxlen=seq_length, truncating='pre')
        encoded_seq = encoded_seq[-seq_length:]
        if len(encoded_seq) < seq_length:
            encoded_seq = [[0]*300]*(seq_length - len(encoded_seq)) + encoded_seq

    # print(len(encoded_seq))
    return np.array(encoded_seq).reshape((1, len(encoded_seq), 300))

def decode_sequence(mapping, text):
    out_text = ""
    for i in range(len(text)):
        # print(mapping.most_similar(positive=[text[i]], topn=5))
        # print(text[i])
        out_text += ' ' + mapping.similar_by_vector(text[i], topn=1)[0][0]

    return out_text
