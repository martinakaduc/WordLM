import os
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sequence_generator import SequenceGenerator
from keras.utils import multi_gpu_model
from text_utils import *
import h5py
import numpy as np

def build_model(vocab_size, seq_length=5, batch_size=32):
    model = Sequential()

    model.add(LSTM(512, return_sequences=True, input_shape=(seq_length, 300)))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(512, activation='linear'))
    model.add(Dense(vocab_size, activation='linear'))

    model.summary()

    return model

def load_multigpu_checkpoint_weights(model, h5py_file):
    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        weight_file = file["model_1"]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]
                print('Loading %s layer...' % layer.name)
            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        if layer.name == 'lstm_1':
                            if term == 'bias:0':
                                weights.insert(0, np.array(layer_weights[term]))
                            elif term =='kernel:0':
                                weights.insert(0, np.array(layer_weights[term]))
                            else:
                                weights.insert(1, np.array(layer_weights[term]))
                        else:
                            weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print(e)
                print("Error: Could not load weights for layer: " + layer.name)

class SaveModel(Callback):
    def __init__(self, ckpt_path='./ckpt', model_path='./model', mode_name='left2right', ckpt_period=1):
        super(SaveModel, self).__init__()
        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.mode_name = mode_name
        self.ckpt_period = ckpt_period

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.ckpt_period == 0:
            self.model.save(os.path.join(self.ckpt_path, 'WordLM_%s_%d.model'%(self.mode_name, epoch)))

    def on_train_end(self, logs=None):
         self.model.save(os.path.join(self.model_path, 'WordLM_%s.model'%self.mode_name))

class WordLM():
    def __init__(self, vocab_size, mapping, seq_length=5, batch_size=32, multi_gpu=False,
                ckpt_path='./ckpt', model_path='./model', mode_name='left2right'):
        self.vocab_size = vocab_size
        self.mapping = mapping
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.mode_name = mode_name
        self.continue_epoch = 0

        if os.path.exists(os.path.join(self.model_path, 'WordLM_%s.model'%self.mode_name)):
            print("Loading saved model...")
            self.model = load_model(os.path.join(self.model_path, 'WordLM_%s.model'%self.mode_name))
        else:
            have_ckpt = self.load_ckpt()
            if not have_ckpt:
                self.model = build_model(self.vocab_size, self.seq_length, self.batch_size)

        if multi_gpu == True:
            self.model = multi_gpu_model(self.model)

    def fit(self, corpus, epochs, ckpt_period=1):
        optimizer = Adam(lr=5e-4, decay=5e-6)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # checkpoint = ModelCheckpoint(os.path.join(self.ckpt_path, 'WordLM_{epoch:03d}.h5'), period=ckpt_period, save_weights_only=True)
        early_stop = EarlyStopping(monitor='loss', patience=50)
        save_model = SaveModel(ckpt_path=self.ckpt_path, model_path=self.model_path, mode_name=self.mode_name, ckpt_period=ckpt_period)
        sequenece_genrator = SequenceGenerator(corpus, self.seq_length, self.mapping, self.vocab_size, batch_size=self.batch_size)

        self.model.fit_generator(generator=sequenece_genrator,
                                epochs=epochs + self.continue_epoch,
                                initial_epoch=self.continue_epoch,
                                callbacks=[save_model, early_stop])

    def get_model(self):
        return self.model

    def get_continue_epoch(self):
        return self.continue_epoch

    def load_ckpt(self):
        ckpt_file = os.listdir(self.ckpt_path)
        ckpt_file = list(filter(lambda x: x[-5:] == 'model', ckpt_file))
        ckpt_file = sorted(ckpt_file)

        if ckpt_file:
            print("Restoring model from checkpoint: %s..."%ckpt_file[-1])
            self.model= load_model(os.path.join(self.ckpt_path, ckpt_file[-1]))
            self.continue_epoch = int(ckpt_file[-1][21:-6]) + 1
            # load_multigpu_checkpoint_weights(self.model, os.path.join(self.ckpt_path, ckpt_file[-1]))
            # self.model.save(os.path.join(self.model_path, 'WordLM_%s.model'%self.mode_name))
            return True
        return False

    def predict(self, X, return_prob_table=False, return_label=True):
        # next_char_predict = self.model.predict_classes(X)
        predict_res = self.model.predict(X)
        if return_prob_table:
            return predict_res

        if return_label:
            next_char_predict = decode_sequence(self.mapping, predict_res)
        return next_char_predict
