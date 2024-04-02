import pickle
import statistics
import librosa as lbs
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import random

with open('dataset.pkl', 'rb') as file:
    dataset:dict = pickle.load(file)

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
def preprocess():
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

def separate_for_training(dataset):
    pieces = list(dataset.keys())
    random.shuffle(pieces)
    train_end_idx = int(0.8 * len(pieces))
    validate_end_idx = int(0.9 * len(pieces))
    train = pieces[:train_end_idx]
    validate = pieces[train_end_idx:validate_end_idx]
    test = pieces[validate_end_idx:]

    return train, validate, test

train, validate, test = separate_for_training(dataset)




'''Takes in name of sample (str) and outputs the labeled progression of either length 32 or length 64
    Parameters:
        midi_seq: string of input name
    Output:
        list with elements being the roots of the chord labels        
'''
# get an array equal to the length of the note matrix where each element is the root note at time n 
def get_progression(midi_seq, dataset):
    seq = dataset[midi_seq]
    roots = np.array(seq['root'])
    roots = roots.flatten()
    roots = [NOTES_NAMES[root] for root in roots]
    return roots


'''Takes in name of sample (str) and outputs the key of the progression
    Parameters:
        midi_seq: string of input name
    Output:
        string with the key of the progression (including either min or major)    
'''
# use tonic and mode info to get key of the progression
def get_key(midi_seq, dataset):
    seq = dataset[midi_seq]
    tonic = seq['tonic']
    mode = seq['mode']
    note_name = NOTES_NAMES[tonic]
    if mode == 'm':
        note_name += 'm'
    return note_name


'''Takes in name of sample (str)
    Parameters:
        midi_seq: string of input name
    Output:
        dictionary with # eigth note as key and 
        a dictionary as the values that has notes as key and amplitude as value
'''
def get_note_map(midi_seq, dataset):
    seq = dataset[midi_seq]
    #get note matrix
    nmat = seq['nmat']
    # get duration from length of progression
    progression = get_progression(midi_seq, dataset)
    len_sequence = len(progression)
    beats = range(len_sequence + 1)
    # create empty map with the # of beats as the key
    note_map = {key:{note:0 for note in NOTES_NAMES} for key in beats}
    for row in nmat:
        start = row[0]
        end = row[1]
        note = librosa.midi_to_note(row[2], unicode=False, octave=False)
        amplitude = row[3]
        for beat in range(start, end + 1):
            note_at_beat = note_map[beat][note]
            # add the amplitude to the current value of the note in the note map
            note_map[beat][note] = note_at_beat + amplitude
    return note_map

# takes in key name of point in dataset and outputs the chroma matrix and chromagram visualization
'''
Takes in name of sample (str) and outputs chromagram plot and dataframe

Parameters:
    midi_seq: name of sample 
Output:
    chromagram plot: heatmap with times as columns, note bins as rows, and values within each box is the amplitude of the note at that time
    
    chromagram dataframe: dataframe with time as the index, note bins as columns with values being their amplitude, and a column at the end that 
    contains the labeled root of the chord at that time 

'''
def get_chromagram(midi_seq):
    # get map of times and note amplitudes at each time
    note_data = get_note_map(midi_seq, dataset)
    # get progression from roots
    progression = get_progression(midi_seq, dataset)
    duration = len(progression)
    # get times from eigth notes -> each note is 0.25 seconds
    times = np.arange(start=0, stop=(duration * 0.25), step=0.25)
    # get the amplitude values of each note bin for all times 
    map = [list(note_data[time].values()) for time in range(duration)] 
    # flip rows and cols -> pitch class as rows and times as cols 
    chroma_map = pd.DataFrame(map).transpose()
    # set notes as rows and times as columns for plotting 
    chroma_map.index = NOTES_NAMES
    chroma_map.index.name = 'Time(s)'
    chroma_map.columns = times
    # get heatmap from df
    # fig, ax = plt.subplots(figsize=(20, 5))
    # sns.heatmap(chroma_map, cmap= 'rocket_r', annot=True, linewidth=0.03, linecolor='black', cbar_kws={'label': 'Amplitude'})
    # make notes from C-Bb instead of Bb-C on plot
    # plt.gca().invert_yaxis()
    # flip df back to having times as rows and chords as columns
    chroma_map = chroma_map.transpose()
    # add labeled chord as a column to dataframe
    chroma_map.insert(12, 'Chord Actual', value=progression)
    return chroma_map

get_chromagram('Niko_Kotoulas_Melody_9_A-Bm-D-G (V-vi-I-IV) - 115-130bpm.mid')

NOTES_NAMES =   ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

## TODO: inital probs, transition probs, emission probs, mu array

def calculate_mu_from_chroma(chroma):
    ''' 
    
    '''
    return chroma[NOTES_NAMES].mean()

def calculate_emission_from_chroma(chroma):
    
    matrices = []

    for chord, group in chroma.groupby('chord'):
        chord_cov_matrix = group[NOTES_NAMES].cov().values
        matrices.append(chord_cov_matrix)

    return np.array(matrices)

def calculate_chord_prob(chord_notes):
    group_count = chord_notes.groupby('following_chords').size().reset_index()
    group_count.columns = ['following_chords', 'count']
    total = group_count.count.sum()
    group_count['transition_probability'] = group_count.count / total
    return group_count

def calculate_transition_probabilites(chroma):
    
    initial_chords = chroma.chord[:-1]
    following_chords = chroma.chord[1:]

    sequence_df = pd.DataFrame({'initial_chords': initial_chords, 'following_chords': following_chords})

    transition_prob_matrix = sequence_df.groupby('initial_chords').apply(calculate_chord_prob).reset_index().drop('level_1', axis=1)

    transition_prob_matrix = transition_prob_matrix.pivot(index='initial_chords', columns='following_chords', values='transition_probability')

    # Transition probabilities for start and end states
    transition_prob_matrix.append(pd.Series(np.zeros(transition_prob_matrix.shape[1]), name='<E>'))

    transition_prob_matrix = transition_prob_matrix.fillna(0)

    transition_prob_matrix['<S>'] = 0
    transition_prob_matrix.loc['<E>'] = 0
    transition_prob_matrix.loc['<E>', '<E>'] = 1

    return transition_prob_matrix
"""
def calculate_initial_probabilities(filenames):
"""