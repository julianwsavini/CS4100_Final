import numpy as np
import pandas as pd
import random
from chroma import *

NOTES_NAMES =   ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FULL_CHORD_LIST = [note + suffix for note in NOTES_NAMES for suffix in ['', 'm', 'dim']]
CUSTOM_ENCODING = {chord: i for i, chord in enumerate(FULL_CHORD_LIST)}
INVERSE_ENCODING = {val: key for key, val in CUSTOM_ENCODING.items()}

def separate_last_chord(chromagram):
    if chromagram.empty:
        return None, chromagram
    last_chord = chromagram.iloc[-1]['Chord Actual']
    chromagram_without_last_chord = chromagram.iloc[:-1]
    return last_chord, chromagram_without_last_chord
"""
Separate preprocessed data into training(80%), validation(10%), and test(10%) sets.
Takes in a the datasdet
Returns a tuple (training, validation, test) of lists of piece names
"""

def separate_for_training(dataset, train_pct):
    pieces = list(dataset.keys())
    random.shuffle(pieces)
    train_end_idx = int(train_pct * len(pieces))
    train = pieces[:train_end_idx]
    test = pieces[train_end_idx:]
    return train, test

def calculate_mu_from_chroma(chroma):
    mu_values = np.zeros(36)
    for i, chord in enumerate(FULL_CHORD_LIST):
        if chord in chroma.columns:
            mu_values[i] = chroma[chord].mean()
        else:
           mu_values[i] = 0
    return pd.DataFrame(pd.Series(mu_values))


def calculate_covariance_from_chroma(chromagram):
    n_features = chromagram.shape[1] - 1  # Assuming last column is 'Chord Actual'
    covariances = np.zeros((len(FULL_CHORD_LIST), n_features, n_features))

    for i, chord in enumerate(FULL_CHORD_LIST):
        chord_segments = chromagram[chromagram['Chord Actual'] == chord].iloc[:, :-1]
        if not chord_segments.empty:
            covariances[i] = np.cov(chord_segments, rowvar=False)
        else:
            covariances[i] = np.eye(n_features)

    return covariances

def calculate_chord_prob(chord_notes):
    group_count = chord_notes.groupby('following_chords').size().reset_index()
    group_count.columns = ['following_chords', 'count']
    total = group_count['count'].sum()
    group_count['transition_probability'] = group_count['count'].astype(np.float64) / float(total)
    return group_count

def calculate_transition_probabilites(chroma):
    initial_chords = chroma['Chord Actual'].values[:-1]
    following_chords = chroma['Chord Actual'][1:].tolist()

    sequence_df = pd.DataFrame({'initial_chords': initial_chords, 'following_chords': following_chords})

    transition_prob_matrix = sequence_df.groupby('initial_chords').apply(calculate_chord_prob).reset_index().drop('level_1', axis=1)

    transition_prob_matrix = transition_prob_matrix.pivot(index='initial_chords', columns='following_chords', values='transition_probability')

    # Initialize a 36x36 DataFrame with zeros
    all_chords_matrix = pd.DataFrame(0., index=FULL_CHORD_LIST, columns=FULL_CHORD_LIST)

    # Update this matrix with the calculated transition probabilities
    all_chords_matrix.update(transition_prob_matrix)

    # Normalize the rows to ensure they sum up to 1
    for chord in all_chords_matrix.index:
        row_sum = all_chords_matrix.loc[chord].sum()
        if row_sum > 0:  # Ensure the row sum is greater than 0 before normalizing
            all_chords_matrix.loc[chord] = all_chords_matrix.loc[chord] / row_sum
        else:
            # Handle rows that sum to 0 here
            all_chords_matrix.loc[chord] = float(1 / 36)
            pass

    all_chords_matrix = all_chords_matrix.div(all_chords_matrix.sum(axis=1), axis=0)

    # Fill any NaN values with 0.
    all_chords_matrix = all_chords_matrix.fillna(0.)

    return all_chords_matrix


def get_initial_chord(file_name, midi_data):
    chords_in_file = get_chord_labels(file_name, midi_data)
    if len(chords_in_file) == 0:
        return None
    chord = get_chord_labels(file_name, midi_data)[0]
    return chord

#returns all initial probabilities, also adapts for dimensions of transition matrix
#returns a 36x1 of probabilities for each chord
def calculate_initial_probabilities(filenames, midi_data):
    first_chords = [get_initial_chord(file_name, midi_data) for file_name in filenames if get_initial_chord(file_name, midi_data) is not None]
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


def predict_next_chords(model, start_chroma, n_predictions):
    current_chroma_df = start_chroma.copy()
    predictions = []
    for _ in range(n_predictions):
        encoded_c = current_chroma_df['Chord Actual'].apply(lambda x: CUSTOM_ENCODING.get(x, -1)).values.reshape(-1, 1)
        next_chord = model.predict(encoded_c)[-1]

        next_chord_name = [chord for chord, encoding in CUSTOM_ENCODING.items() if encoding == next_chord][0]
        predictions.append(next_chord_name)
        #update chroma for next round
        new_row = pd.DataFrame([[next_chord_name]], columns=['Chord Actual'])
        current_chroma_df = pd.concat([current_chroma_df, new_row], ignore_index=True)

    return predictions



def chord_distance_with_quality(chord1, chord2):
    note_to_semitone = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
    }
    quality_weight = {
        ('', ''): 0,
        ('', 'm'): 1, ('m', ''): 1,
        ('', 'dim'): 2, ('dim', ''): 2,
        ('m', 'dim'): 1, ('dim', 'm'): 1,
    }

    root1, quality1 = chord1[:2] if chord1[1:2] == '#' else chord1[0], chord1[2:]
    root2, quality2 = chord2[:2] if chord2[1:2] == '#' else chord2[0], chord2[2:]

    pos1 = note_to_semitone.get(root1, 0)
    pos2 = note_to_semitone.get(root2, 0)

    semitone_distance = min(pos1 - pos2, 12 - (pos1 - pos2))
    quality_distance = quality_weight.get((quality1, quality2), 2)  # Default to 2 if combination not in weight dictionary
    distance = semitone_distance + quality_distance

    return distance

def mean_chord_distance_with_quality(predicted, actual):
    # predicted and actual are arrays that
    # each hold chords in order where each indedx holds matching items
    total_distance = 0
    num_pairs = len(predicted)

    for i in range(num_pairs):
        distance = chord_distance_with_quality(predicted[i], actual[i])
        total_distance += distance

    avg_distance = total_distance / num_pairs if num_pairs > 0 else 0
    return avg_distance