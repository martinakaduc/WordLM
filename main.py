import os
import argparse
import pickle
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sequence_generator import SequenceGenerator
from model import WordLM, SaveModel
from text_utils import *
from file_utils import *
from gensim import models

def main(args):
    if os.path.exists(args.corpus[:-4]+'_processed.txt'):
        raw_text = load_data(args.corpus[:-4]+'_processed.txt', processed=True)
    else:
        raw_text = load_data(args.corpus)
        # raw_text = text_cleaner(raw_text)
        with open(args.corpus[:-4]+'_processed.txt', 'w', encoding='utf8') as f:
            f.write(raw_text)

    mapping = models.KeyedVectors.load('word2vec_skipgram.bin')
    ########################################
    raw_text = raw_text.split()
    # print(raw_text)
    vocab_size = 300 # Now it word embed size
    generic_lm = WordLM(vocab_size, mapping, seq_length=args.seq_length, multi_gpu=args.multi_gpu,
                batch_size=args.batch_size, ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode)
    ########################################

    if args.low_ram:
        if args.mode == 'right2left':
            raw_text = raw_text[::-1]

        model = generic_lm.get_model()
        continue_epoch = generic_lm.get_continue_epoch()

        optimizer = Adam(lr=5e-4, decay=5e-6)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # checkpoint = ModelCheckpoint(os.path.join(args.ckpt_path, 'WordLM_{epoch:03d}.h5'), period=args.ckpt_period)
        early_stop = EarlyStopping(monitor='loss', patience=12)
        save_model = SaveModel(ckpt_path=args.ckpt_path, model_path=args.model_path, mode_name=args.mode, ckpt_period=args.ckpt_period)
        sequenece_genrator = SequenceGenerator(raw_text, args.seq_length, mapping, vocab_size, batch_size=args.batch_size)

        model.fit_generator(generator=sequenece_genrator,
                                epochs=args.epochs + continue_epoch,
                                initial_epoch=continue_epoch,
                                callbacks=[save_model, early_stop])

        model.save(os.path.join(args.model_path, 'WordLM_%s.model'%args.mode))

    else:
        if args.mode == 'right2left':
            raw_text = raw_text[::-1]

        generic_lm.fit(raw_text, epochs=args.epochs, ckpt_period=args.ckpt_period)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='truyen_kieu.txt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='left2right')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--ckpt_period', type=int, default=1)
    parser.add_argument('--low_ram', type=bool, default=True)

    args = parser.parse_args()

    assert args.mode in ['left2right', 'right2left'], "Choose one of these mode: left2right, right2left."
    main(args)
