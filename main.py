"""
Author: Mohammad Ramezani
Created: September 30, 2022
"""

import collections
import os
import random
import shutil

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

DIRECTORY = 'D:/Business/Idea_Music/Data/Original_Data/IMSLP_GenMusicSeq2Seq/'
COMPLEXITY_LEVEL_NUMBER = 12

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        cnt = len(notes['pitch'])
        title = f'First {count} of {cnt} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])

    plt.figure()
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    plt.title(title)
    plt.show()
    print()


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x="pitch", bins=20)

    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))

    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))
    plt.show()


def notes_to_midi(notes: pd.DataFrame,
                  out_file: str,
                  instrument_name: str,
                  velocity: int = 100,  # note loudness
                  ) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(velocity=velocity,
                                pitch=int(note['pitch']),
                                start=start,
                                end=end)
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)

    return pm


def create_sequences(dataset: tf.data.Dataset,
                     seq_length: int,
                     vocab_size=128,
                     ) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    # Mean squared error
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)

    return tf.reduce_mean(mse + positive_pressure)  # Computes the mean of elements across dimensions of a tensor


def predict_next_note(notes: np.ndarray,
                      keras_model: tf.keras.Model,
                      temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


if __name__ == '__main__':

    # # -------------------------------------------------------------------------------------------
    # # Loading Data + Feature Extraction
    # # -------------------------------------------------------------------------------------------
    #
    # for cnt_complexity in range(1, COMPLEXITY_LEVEL_NUMBER + 1):
    #     all_notes = []
    #     print(cnt_complexity)
    #     temp_dir = DIRECTORY + "{:02d}".format(cnt_complexity)
    #     dir_list = os.listdir(temp_dir)
    #     # dir_list = [dir_list[0]]  # to choose a desired sample
    #     cnt_file = 0
    #     for filename in dir_list:
    #         if filename[-3:] == 'mid':
    #             cnt_file += 1
    #             midi_file_path = temp_dir + '/' + filename
    #
    #             pm = pretty_midi.PrettyMIDI(midi_file_path)
    #
    #             # some inspection on the MIDI file
    #             # print('Number of instruments:', len(pm.instruments))
    #             instrument = pm.instruments[0]
    #             instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    #             # print('Instrument name:', instrument_name)
    #
    #             # # Extract notes
    #             # # three variables to represent a note when training the model: pitch, step and duration.
    #             # # The pitch is the perceptual quality of the sound as a MIDI note number.
    #             # # The step is the time elapsed from the previous note or start of the track.
    #             # # The duration is how long the note will be playing in seconds. (note end - start time).
    #             # print()
    #             # for i, note in enumerate(instrument.notes[:10]):
    #             #     note_name = pretty_midi.note_number_to_name(note.pitch)
    #             #     duration = note.end - note.start
    #             #     print(f'{i}: pitch={note.pitch}, note_name={note_name},'
    #             #           f' duration={duration:.4f}')
    #             #
    #             # raw_notes = midi_to_notes(midi_file_path)
    #             #
    #             # get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    #             # sample_note_names = get_note_names(raw_notes['pitch'])
    #             #
    #             # # Visualizing the musical piece, Pianoroll
    #             # count = len(raw_notes['pitch'])
    #             # print('\nSample note numbers:', count)
    #             # plot_piano_roll(raw_notes, count=30)
    #             # plot_piano_roll(raw_notes)
    #             #
    #             # # Note Distribution:
    #             # plot_distributions(raw_notes)
    #             #
    #             # # Generate MIDI file
    #             # example_file = 'example.midi'
    #             # example_pm = notes_to_midi(raw_notes, out_file=example_file, instrument_name=instrument_name)
    #
    #             # Create the training dataset
    #             notes = midi_to_notes(midi_file_path)
    #             all_notes.append(notes)
    #
    #     all_notes = pd.concat(all_notes)
    #     n_notes = len(all_notes)
    #     print('Number of notes parsed for class:', n_notes)
    #
    #     # Creating a TF
    #     key_order = ['pitch', 'step', 'duration']
    #     train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    #     notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    #     print(notes_ds.element_spec)
    #
    #     # Training the model on batches of sequences of notes.
    #     # In this way, the model will be trained to predict the next note in a sequence.
    #     seq_length = 25
    #     vocab_size = 128
    #     seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    #     print(seq_ds.element_spec)
    #
    #     for seq, target in seq_ds.take(1):
    #         print('sequence shape:', seq.shape)
    #         print('sequence elements (first 10):', seq[0: 10])
    #         print()
    #         print('target:', target)
    #
    #     batch_size = 64
    #     buffer_size = n_notes - seq_length  # the number of items in the dataset
    #     train_ds = (seq_ds.shuffle(buffer_size)
    #                 .batch(batch_size, drop_remainder=True)
    #                 .cache()
    #                 .prefetch(tf.data.experimental.AUTOTUNE))
    #     print(train_ds.element_spec)
    #
    #     # Create and train the model
    #     input_shape = (seq_length, 3)
    #     learning_rate = 0.005
    #
    #     inputs = tf.keras.Input(input_shape)  # to instantiate a Keras tensor
    #     x = tf.keras.layers.LSTM(128)(inputs)  # Units: 128, inputs: A 3D tensor with shape [batch, timesteps, feature]
    #
    #     outputs = {
    #         'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    #         'step': tf.keras.layers.Dense(1, name='step')(x),
    #         'duration': tf.keras.layers.Dense(1, name='duration')(x),
    #     }
    #
    #     model = tf.keras.Model(inputs, outputs)
    #
    #     loss = {
    #         'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         'step': mse_with_positive_pressure,
    #         'duration': mse_with_positive_pressure,
    #     }
    #
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #     model.compile(loss=loss, optimizer=optimizer)
    #
    #     print(model.summary())
    #
    #     losses = model.evaluate(train_ds, return_dict=True)
    #     print(losses)
    #
    #     model.compile(loss=loss,
    #                   loss_weights={'pitch': 0.05,
    #                                 'step': 1.0,
    #                                 'duration': 1.0},
    #                   optimizer=optimizer)
    #
    #     model.evaluate(train_ds, return_dict=True)
    #
    #     callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}',
    #                                                     save_weights_only=True),
    #                  # to save the Keras model or model weights at some frequency
    #                  tf.keras.callbacks.EarlyStopping(monitor='loss',
    #                                                   patience=5,
    #                                                   verbose=1,
    #                                                   restore_best_weights=True)]
    #
    #     epochs = 100
    #     history = model.fit(train_ds,
    #                         epochs=epochs,
    #                         callbacks=callbacks)
    #
    #     model_path = "./output/RNN_midi_" + "{:02d}".format(cnt_complexity) + ".h5"
    #     model.save(model_path)
    #
    #     plt.figure()
    #     plt.plot(history.epoch, history.history['loss'], label='total loss')
    #     lossplot_path = "./output/RNN_midi_" + "{:02d}".format(cnt_complexity) + ".png"
    #     plt.savefig(lossplot_path, bbox_inches='tight')

    print()

    # -------------------------------------------------------------------------------------------
    # Generate MIDI
    # -------------------------------------------------------------------------------------------
    # We will first need to provide a starting sequence of notes.
    # The function generates one note from a sequence of notes.

    # For note pitch, it draws a sample from the softmax distribution of notes produced by the model,
    # and does not simply pick the note with the highest probability. Always picking the note with the highest
    # probability would lead to repetitive sequences of notes being generated.

    key_order = ['pitch', 'step', 'duration']

    # The temperature parameter can be used to control the randomness of notes generated.
    # to adjust the temperature parameter to generate more or less random predictions.
    temperature = 10
    num_predictions = 120
    seq_length = 25
    vocab_size = 128

    data_dir = 'D:/Business/Idea_Music/Data/Original_Data/IMSLP_GenMusicSeq2Seq/'
    for cnt_complexity in range(1, COMPLEXITY_LEVEL_NUMBER + 1):

        os.mkdir('./sample_from_class_{:02d}'.format(cnt_complexity))

        # Select random file from each directory
        sample_dir = data_dir + os.listdir(data_dir)[cnt_complexity - 1]
        sample_path = sample_dir + '/' + random.choice(os.listdir(sample_dir))

        pm = pretty_midi.PrettyMIDI(sample_path)
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        raw_notes = midi_to_notes(sample_path)
        sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

        # The initial sequence of notes; pitch is normalized similar to training sequences
        input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

        for cnt_difficulty in range(1, COMPLEXITY_LEVEL_NUMBER + 1):

            print(cnt_difficulty)
            model_path = "./output/RNN_midi_" + "{:02d}".format(cnt_difficulty) + ".h5"
            model = keras.models.load_model(model_path, custom_objects={"mse_with_positive_pressure":
                                                                            mse_with_positive_pressure})

            generated_notes = []
            prev_start = 0
            for _ in range(num_predictions):
                pitch, step, duration = predict_next_note(input_notes, model, temperature)
                start = prev_start + step
                end = start + duration
                input_note = (pitch, step, duration)
                generated_notes.append((*input_note, start, end))
                input_notes = np.delete(input_notes, 0, axis=0)
                input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
                prev_start = start

            generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

            shutil.copy(sample_path,
                        'D:/Business/Idea_Music/Code/GenerateMusic_RNN/sample_from_class_{:02d}/'
                        .format(cnt_complexity))

            out_file = './sample_from_class_{:02d}/generated_from_model_{:02d}.mid'.format(cnt_complexity,
                                                                                           cnt_difficulty)
            out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)
