"""functions to get input chromagrams """
import numpy as np
import pandas as pd
import pickle
import librosa
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy.ndimage import gaussian_filter

with open(r"dataset.pkl", 'rb') as data:
    midi_data = pickle.load(data)

notes =  ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FULL_CHORD_LIST = [note + suffix for note in notes for suffix in ['', 'm', 'dim']]

'''Takes in name of sample (str) and outputs the labeled progression of either length 32 or length 64
    Parameters:
        midi_seq: string of input name
    Output:
        list with elements being the roots of the chord labels        
'''
# get an array equal to the length of the note matrix where each element is the root note at time n 
def get_progression(midi_seq, midi_data):
    seq = midi_data[midi_seq]
    roots = np.array(seq['root'])
    roots = roots.flatten()
    roots = [notes[root] for root in roots]
    return roots


'''Takes in name of sample (str) and outputs the key of the progression
    Parameters:
        midi_seq: string of input name
    Output:
        string with the key of the progression (including either min or major)    
'''
# use tonic and mode info to get key of the progression
def get_key(midi_seq, midi_data):
    seq = midi_data[midi_seq]
    tonic = seq['tonic']
    mode = seq['mode']
    note_name = notes[tonic]
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
def get_note_map(midi_seq, midi_data):
    seq = midi_data[midi_seq]
    #get note matrix
    nmat = seq['nmat']
    # get duration from length of progression
    progression = get_progression(midi_seq, midi_data)
    len_sequence = len(progression)
    beats = range(len_sequence + 1)
    # create empty map with the # of beats as the key
    note_map = {key:{note:0 for note in notes} for key in beats}
    for row in nmat:
        start = row[0]
        end = row[1]
        note = librosa.midi_to_note(row[2], unicode=False, octave=False)
        amplitude = row[3]
        for beat in range(start, end):
            note_at_beat = note_map[beat][note]
            # add the amplitude to the current value of the note in the note map
            note_map[beat][note] = note_at_beat + amplitude
    return note_map

'''
Function that parses the label to get prog info and gets the true relabeled progression for a given piece
    Params:
        -sample: sample name(str)
        -midi_data: midi dataset(dict)
    Outputs:
        -pandas series with the correctly labeled progression for sequence 
'''
def get_chord_labels(sample, midi_data):
    progression = pd.Series(get_progression(sample, midi_data))
    roots_in_seq = list(set(progression))
    new_prog = progression.copy()
    # split samples by '_' and '-' characters 
    for string in sample.split('_'):
        if '-' in string:
            prog = string.split('-')
            prog = (' ').join(prog).split(' ')
            # filter out chords not in prog list 
            chords = [chord for chord in prog if chord in FULL_CHORD_LIST]
            if len(chords) > 0:
                labels_to_string = (' ').join(chords)
                for chord in roots_in_seq:
                    if chord not in chords:
                        # define regex pattern to get an instance of the chord 
                        pattern = r'\b\w*{}\w*\b'.format(re.escape(chord))
                        # replace the root labels with the true 
                        matches = re.findall(pattern, labels_to_string, flags=re.IGNORECASE)
                        if len(matches) == 0:
                            continue
                        true_chord = matches[0]
                        new_prog = new_prog.replace(to_replace=chord, value=true_chord)
    return new_prog 

'''
Takes in name of sample (str) and outputs chromagram data

Parameters:
    midi_seq: name of sample 
Output:
    chromagram dataframe: dataframe with time as the index, note bins as columns with values being their amplitude, and a column at the end that 
    contains the labeled root of the chord at that time 
'''
def get_chromagram(midi_seq, midi_data):
    # get map of times and note amplitudes at each time
    note_data = get_note_map(midi_seq, midi_data)
    # get progression from roots
    progression = get_progression(midi_seq, midi_data)
    duration = len(progression)
    # get the amplitude values of each note bin for all times 
    map = [list(note_data[time].values()) for time in range(duration)] 
    # flip rows and cols -> pitch class as rows and times as cols 
    chroma_map = pd.DataFrame(map, columns=notes)
    chroma_map['Chord Actual'] = get_chord_labels(midi_seq, midi_data)
    # filter out sequences that have nothing at end 
    chroma_map = chroma_map.loc[(chroma_map.loc[:, notes] != 0).any(axis=1)]
    return chroma_map

'''
Takes in name of sample (str) and outputs chromagram plot

Parameters:
    midi_seq: name of sample 
Output:
    chromagram plot: heatmap with times as columns, note bins as rows, and values within each box is the amplitude of the note at that time
'''
def get_chromagram_plot(midi_seq, midi_data):
    # get map of times and note amplitudes at each time
    note_data = get_note_map(midi_seq, midi_data)
    # get progression from roots
    progression = get_progression(midi_seq, midi_data)
    duration = len(progression)
    # get times from eigth notes -> each note is 0.25 seconds
    times = np.arange(start=0, stop=(duration * 0.25), step=0.25)
    # get the amplitude values of each note bin for all times 
    map = [list(note_data[time].values()) for time in range(duration)] 
    # flip rows and cols -> pitch class as rows and times as cols 
    chroma_map = pd.DataFrame(map).transpose()
    # set notes as rows and times as columns for plotting 
    chroma_map.index = notes
    chroma_map.index.name = 'Time(s)'
    chroma_map.columns = times
    # Drop columns where all rows have 0
    chroma_map = chroma_map.loc[:, (chroma_map != 0).any(axis=0)]
    # get heatmap from df
    fig, ax = plt.subplots(figsize=(20, 5))
    ax = sns.heatmap(chroma_map, cmap= 'rocket_r', linewidth=0.03, linecolor='black', cbar_kws={'label': 'Amplitude'})
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Note')
    ax.set_title(midi_seq)
    plt.gca().invert_yaxis()
    plt.show()