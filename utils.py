import pickle
import statistics
import librosa as lbs
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import random
from chroma import *


midi_to_note_dict = {}

def midi_to_note(midi_number):
    if midi_number not in midi_to_note_dict:
        midi_to_note_dict[midi_number] = lbs.midi_to_note(midi_number, unicode=False)
    return midi_to_note_dict[midi_number]


"""
Returns a dictionary with {piece name, details dict} 
    dictionaries for every piece {time, (mean amplitude, [notes at that point])}
We choose to keep the piece names here so we can later find mode or other attributes 
using dataset[piece_name]['mode']
"""
def preprocess(filename):

    with open(filename, 'rb') as file:
        dataset:dict = pickle.load(file)

    all_pieces = {}
    for piece_name in tqdm(list(dataset.keys())):
        notes_per_time = {}
        piece = dataset[piece_name]
        nmat = sorted(piece['nmat'], key=lambda note: note[0])
        piece_length = nmat[len(nmat)-1][1] #last time of last note playing
        for i in range(0, piece_length+1): #for each second, what notes are playing?
            amplitudes = []
            notes = []
            for note in nmat:
                if i>=note[0] and i<=note[1]: #it's in the range when the note plays
                    amplitudes.append(note[3])
                    notes.append(midi_to_note(note[2]))
                elif note[0]>i: #once the start time is after i, break
                    break
            mean_amp = 0 if amplitudes==[] else statistics.mean(amplitudes)
            notes_per_time[i] = (mean_amp, notes)
        all_pieces[piece_name] = notes_per_time
    return all_pieces

#pieces_beats = preprocess()

"""
For validation/testing, separates a given piece into x last beats and len-x previous notes/chords.
Takes in a dictionary {beat, (mean amplitude, [notes at that point])}
and x=number of beats to separate form the end

Returns tuple of two dicts: (dict_with_last_x_items, dict_with_the_rest)
"""
def separate_last_note_group(piece, x):
    if x == 0 or x < 0:
        return {}, piece
    items = list(piece.items())
    dict_with_last_x_items = dict(items[-x:])
    dict_with_the_rest = dict(items[:-x])
    return dict_with_last_x_items, dict_with_the_rest

"""
Separate preprocessed data into training(80%), validation(10%), and test(10%) sets.
Takes in a the datasdet
Returns a tuple (training, validation, test) of lists of piece names
"""

def separate_for_training(dataset, train_pct, val_pct):
    pieces = list(dataset.keys())
    random.shuffle(pieces)
    train_end_idx = int(train_pct * len(pieces))
    validate_end_idx = int((train_pct + val_pct) * len(pieces))
    train = pieces[:train_end_idx]
    validate = pieces[train_end_idx:validate_end_idx]
    test = pieces[validate_end_idx:]

    return train, validate, test

# train, validate, test = separate_for_training(dataset, .8, .1)


NOTES_NAMES =   ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


## TODO: inital probs, transition probs, emission probs, mu array

def calculate_mu_from_chroma(chroma):
    ''' 
    
    '''
    return chroma[NOTES_NAMES].mean()

def calculate_emission_from_chroma(chroma):
    
    matrices = []

    for chord, group in chroma.groupby('Chord Actual'):
        chord_cov_matrix = group[NOTES_NAMES].cov().values
        matrices.append(chord_cov_matrix)

    return np.array(matrices)

def calculate_chord_prob(chord_notes):
    group_count = chord_notes.groupby('following_chords').size().reset_index()
    group_count.columns = ['following_chords', 'count']
    total = group_count['count'].sum()
    group_count['transition_probability'] = group_count['count'] / total
    return group_count

def calculate_transition_probabilites(chroma):
    
    initial_chords = chroma['Chord Actual'].values[:-1]
    following_chords = chroma['Chord Actual'][1:].tolist()

    sequence_df = pd.DataFrame({'initial_chords': initial_chords, 'following_chords': following_chords})

    transition_prob_matrix = sequence_df.groupby('initial_chords').apply(calculate_chord_prob).reset_index().drop('level_1', axis=1)

    transition_prob_matrix = transition_prob_matrix.pivot(index='initial_chords', columns='following_chords', values='transition_probability')

    # Transition probabilities for start and end states
    transition_prob_matrix['<E>'] = pd.Series(np.zeros(transition_prob_matrix.shape[1]), name='<E>')

    transition_prob_matrix = transition_prob_matrix.fillna(0)

    transition_prob_matrix['<S>'] = 0
    transition_prob_matrix.loc['<E>'] = 0
    transition_prob_matrix.loc['<E>', '<E>'] = 1

    return transition_prob_matrix
"""
def calculate_initial_probabilities(filenames):
"""

def predict(pcp, model, mu):
    """
    :param pcp: chroma
    :param model: the hmm model
    :return: the pcp, with a column with the predicted chords
    """
    preds = model.predict(pcp)

    chords = mu.index.values
    chord_idxs = range(len(mu.index.values))

    map = {chord_num: chord_letter for chord_num, chord_letter in zip(chord_idxs, chords)}

    preds_str = np.array([map[chord_ix] for chord_ix in preds])

    pcp['predictions'] = preds_str

    return pcp


def get_unique_predicted(pcp):
    """
    :param pcp: chroma, with predicted column
    :return: filtered preds, a filtered chroma with note changes
    """
    predictions_list = pcp['predictions'].tolist()

    # Initialize an empty list to store the boolean values
    chord_unique = []

    # Iterate over the list, comparing each element with the previous one
    for i in range(len(predictions_list)):
        if i == 0:
            chord_unique.append(True)
        else:
            if predictions_list[i] != predictions_list[i - 1]:
                chord_unique.append(True)
            else:
                chord_unique.append(False)

    chord_unique = pd.Series(chord_unique)
    chord_unique_idxs = chord_unique[chord_unique == True].index
    filtered_preds = pcp.loc[chord_unique_idxs].copy()
    filtered_preds['start'] = np.array([0] + filtered_preds['end'][:-1].tolist())
    return filtered_preds[['predictions', 'start', 'end']]