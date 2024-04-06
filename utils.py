import pickle
import statistics
import librosa as lbs
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from chroma import *

NOTES_NAMES =   ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FULL_CHORD_LIST = [note + suffix for note in NOTES_NAMES for suffix in ['', 'm', 'dim']]

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

def calculate_mu_from_chroma(chroma):
    mu_values = np.zeros(36)
    for i, chord in enumerate(FULL_CHORD_LIST):
        if chord in chroma.columns:
            mu_values[i] = chroma[chord].mean()
        else:
           mu_values[i] = 0
    return mu_values

def calculate_emission_from_chroma(chroma):
    emission_matrices = np.zeros((36,36,36))
    chord_counts = chroma['Chord Actual'].value_counts(normalize=True)

    for i, chord in enumerate(FULL_CHORD_LIST):
        matrix = np.zeros((36,36))
        # If the chord is observed in the dataset, fill its matrix diagonally
        if chord in chord_counts:
            np.fill_diagonal(matrix, chord_counts[chord])
        else:
            # For chords not present in the dataset, consider a minimal presence value
            np.fill_diagonal(matrix, 0.01)
        emission_matrices[i] = matrix
    return emission_matrices

def calculate_chord_prob(chord_notes):
    group_count = chord_notes.groupby('following_chords').size().reset_index()
    group_count.columns = ['following_chords', 'count']
    total = group_count['count'].sum()
    group_count['transition_probability'] = group_count['count'].astype(np.float64) / float(total)
    return group_count

def calculate_transition_probabilites(chroma):
    # Look into splitting between songs somehow
    initial_chords = chroma['Chord Actual'].values[:-1]
    following_chords = chroma['Chord Actual'][1:].tolist()

    sequence_df = pd.DataFrame({'initial_chords': initial_chords, 'following_chords': following_chords})

    transition_prob_matrix = sequence_df.groupby('initial_chords').apply(calculate_chord_prob).reset_index().drop('level_1', axis=1)

    transition_prob_matrix = transition_prob_matrix.pivot(index='initial_chords', columns='following_chords', values='transition_probability')

    # Initialize a 36x36 DataFrame with zeros
    all_chords_matrix = pd.DataFrame(0, index=FULL_CHORD_LIST, columns=FULL_CHORD_LIST)

    # Update this matrix with the calculated transition probabilities
    all_chords_matrix.update(transition_prob_matrix)

    # Normalize the rows to ensure they sum up to 1
    for chord in all_chords_matrix.index:
        row_sum = all_chords_matrix.loc[chord].sum()
        if row_sum > 0:  # Ensure the row sum is greater than 0 before normalizing
            all_chords_matrix.loc[chord] = all_chords_matrix.loc[chord] / row_sum
        else:
            # Handle rows that sum to 0 here
            all_chords_matrix.loc[chord] = 1 / 36
            pass

    all_chords_matrix = all_chords_matrix.div(all_chords_matrix.sum(axis=1), axis=0)

    # Fill any NaN values with 0
    all_chords_matrix = all_chords_matrix.fillna(0)

    return all_chords_matrix


def get_initial_chord(file_name, midi_data):
    mode = midi_data[file_name]['mode']
    # check if sequence is in a minor or major scale
    if mode == 'm':
        seq_scale = get_minor_scale(NOTES_NAMES, file_name, midi_data)
    else:
        seq_scale = get_maj_scale(NOTES_NAMES, file_name, midi_data)
    # get the first chord
    chord = list(get_progression(file_name, midi_data))[0]
    if chord not in seq_scale:
        # define regex pattern to get an instance of the chord
        pattern = r'\b\w*{}\w*\b'.format(re.escape(chord))
        # join list of chords into a single string to parse for regex and replace in chroma labels
        # each seq_scale is of length 7
        scale_to_string = (' ').join(seq_scale)
        # find true chord
        found_chords = re.findall(pattern, scale_to_string, flags=re.IGNORECASE)
        if found_chords:
            chord = found_chords[0]
        else:
            return None
    return chord

#returns all initial probabilities, also adapts for dimensions of transition matrix
#returns a 36x1 of probabilities for each chord
def calculate_initial_probabilities(filenames, midi_data):
    first_chords = []
    # Get all initial chords
    for file_name in filenames:
        chord = get_initial_chord(file_name, midi_data)
        if chord is not None:
            first_chords.append(chord)
    chord_counts = np.unique(first_chords, return_counts=True)
    total_num_chords = chord_counts[1].sum()
    # Create a Series from the counts
    probabilities = chord_counts[1].astype(np.float64)/float(total_num_chords)
    initial_probs = pd.Series(probabilities, index=chord_counts[0])
    all_chords = pd.Series(np.zeros(len(FULL_CHORD_LIST)), index=FULL_CHORD_LIST)
    all_chords.update(initial_probs)
    diff = 1.0 - all_chords.sum()
    if diff != 0:
        max_index = np.argmax(all_chords)
        all_chords.iloc[max_index] += diff

    return all_chords

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


def new_predict(chroma, model):
    """
    Predict chord labels for each observation in chroma using the given HMM model.

    :param chroma: DataFrame or 2D numpy array with chroma features for prediction.
    :param model: Trained HMM model for chord prediction.
    :param FULL_CHORD_LIST: List of chord names corresponding to model states, in the same order as model states.
    :return: chroma with an additional column 'predictions' containing predicted chords.
    """
    # Making predictions using the HMM model
    preds = model.predict(chroma)

    # Mapping predicted state indices to chord names using FULL_CHORD_LIST
    preds_str = np.array([FULL_CHORD_LIST[state] for state in preds])

    # Adding predictions back to the input DataFrame or numpy array
    if isinstance(chroma, pd.DataFrame):
        chroma_with_preds = chroma.copy()
        chroma_with_preds['predictions'] = preds_str
    else:
        # If chroma is a numpy array, concatenate the predictions as a new column
        # Note: This requires chroma to be 2D and preds_str to be reshaped as a column vector
        chroma_with_preds = np.hstack([chroma, preds_str.reshape(-1, 1)])

    return chroma_with_preds


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

def format_indiv_chroma(unformatted_chroma:pd.DataFrame):
    # TODO: determine what values go in the start and end rows
    zeroed_vals = [[0 for i in range(unformatted_chroma.shape[1])]]
    start = pd.DataFrame([zeroed_vals[0][:-1] + [['<S>']]], columns=unformatted_chroma.columns)
    middle = unformatted_chroma
    end = pd.DataFrame([zeroed_vals[0][:-1] + [['<E>']]], columns=unformatted_chroma.columns)


    formatted_chroma = pd.concat([start, middle, end]).reset_index(drop=True)
    return formatted_chroma