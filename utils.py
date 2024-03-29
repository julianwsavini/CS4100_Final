import pickle
import statistics
import librosa as lbs
from tqdm import tqdm

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

pieces_beats = preprocess()

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
Takes in a dict {piece_name, details} where details is a dict {time, (mean amplitude, [notes at that point])}
Returns a tuple (training, validation, test)
"""
def separate_for_training(all_pieces):
    pieces = list(all_pieces.items())
    train_end_idx = int(0.8 * len(pieces))
    validate_end_idx = int(0.9 * len(pieces))

    train = dict(pieces[:train_end_idx])
    validate = dict(pieces[train_end_idx:validate_end_idx])
    test = dict(pieces[validate_end_idx:])

    return train, validate, test

train, validate, test = separate_for_training(pieces_beats)
