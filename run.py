import argparse
import pickle
from model import WordLM
from text_utils import *
from gensim import models

def main(args):
    mapping = models.KeyedVectors.load('word2vec_skipgram.bin')
    vocab_size = 300

    generic_lm = WordLM(vocab_size, mapping, seq_length=args.seq_length, multi_gpu=args.multi_gpu,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)

    while True:
        input_text = input('Input: ')
        if input_text.lower() == 'exit':
            break

        if (args.mode == 'right2left'):
            input_text = input_text[::-1]

        for _ in range(args.predict_length):
            encoded_text = encode_sequence(mapping, input_text.lower().split(), args.seq_length, padding=True)
            # print(encoded_text)
            next_char = generic_lm.predict(encoded_text, return_prob_table=False, return_label=True)
            input_text += next_char

        if (args.mode == 'right2left'):
            input_text = input_text[::-1]

        print('Predict: %s\n' % input_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='left2right')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--predict_length', type=int, default=10)

    args = parser.parse_args()

    assert args.mode in ['left2right', 'right2left'], "Choose one of these mode: left2right, right2left."
    main(args)
