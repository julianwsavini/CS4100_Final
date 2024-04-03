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


'''Takes in name of sample (str) and outputs the labeled progression of either length 32 or length 64
    Parameters:
        midi_seq: string of input name
    Output:
        list with elements being the roots of the chord labels        
'''
# get an array equal to the length of the note matrix where each element is the root note at time n 
def get_progression(midi_seq):
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
def get_key(midi_seq):
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
def get_note_map(midi_seq):
    seq = midi_data[midi_seq]
    #get note matrix
    nmat = seq['nmat']
    # get duration from length of progression
    progression = get_progression(midi_seq)
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
Function to rotate the list of notes so that the first element becomes the root and the following elements are the consecutive chords rotated for that scale

Parameters: 
    original notes C through B (list)
    tonic from midi sequence data (int)

Output:
    rotated list of note strings
'''
# change the indices of the notes so that the key will be the first chord 
def rotate_list(notes, tonic):
    try:
        # get root note of first chord in prog
        key_root = notes[tonic]
        # get the curr index of the root note in notes list 
        index = notes.index(key_root)
    except ValueError:
        print(f"{key_root} not found in the list.")
        return notes
    # use splicing and concatonation to rotate the notes list so that there are still 12 elements but the first element and preceding elements are flipped for the given key 
    rotated_lst = notes[index:] + notes[:index]
    return rotated_lst

'''
Function that takes in string of sample name with a major mode
and outputs a list of the chords in that key -> use for label realignment

Parameters:
    midi sample name (str)
    
Output:
    chords in the key of the provided sequence name
'''
def get_maj_scale(notes, midi_seq):
    seq = midi_data[midi_seq]
    key_chords = []
    # get key info
    tonic = seq['tonic']
    notes_rotated = rotate_list(notes, tonic)
    '''
    1) tonic, root (major)
    2) 2 semitones up (minor)
    3) 2 semitones up (minor)
    4) 1 semitone up (major)
    5) 2 semitones up (major)
    6) 2 semitones up (minor)
    7) 2 semitones up (diminished)
    '''
    key_chords.append(notes_rotated[0])
    key_chords.append(notes_rotated[2] + 'm')
    key_chords.append(notes_rotated[4] + 'm')
    key_chords.append(notes_rotated[5])
    key_chords.append(notes_rotated[7])
    key_chords.append(notes_rotated[9] + 'm')
    key_chords.append(notes_rotated[11] + 'dim')
    return key_chords

'''
Function that takes in string of sample name with a minor mode
and outputs a list of the chords in that key 

Parameters:
    midi sample name (str)
    
Output:
    chords in the key of the provided sequence name
'''
def get_minor_scale(notes, midi_seq):
    seq = midi_data[midi_seq]
    key_chords = []
    # get key info
    tonic = seq['tonic']
    notes_rotated = rotate_list(notes, tonic)
    '''
    1) tonic, root (minor)
    2) 2 semitones up (diminished)
    3) 1 semitone up (major)
    4) 2 semitones up (minor)
    5) 2 semitones up (minor)
    6) 1 semitones up (major)
    7) 2 semitones up (major)
    '''
    key_chords.append(notes_rotated[0] + 'm')
    key_chords.append(notes_rotated[2] + 'dim')
    key_chords.append(notes_rotated[3])
    key_chords.append(notes_rotated[5] + 'm')
    key_chords.append(notes_rotated[7] + 'm')
    key_chords.append(notes_rotated[8])
    key_chords.append(notes_rotated[10])
    return key_chords


'''
Takes in name of sample (str) and outputs chromagram data

Parameters:
    midi_seq: name of sample 
Output:
    chromagram dataframe: dataframe with time as the index, note bins as columns with values being their amplitude, and a column at the end that 
    contains the labeled root of the chord at that time 
'''

def get_chromagram(midi_seq):
    # get map of times and note amplitudes at each time
    note_data = get_note_map(midi_seq)
    # get progression from roots
    progression = get_progression(midi_seq)
    duration = len(progression)
    # get the amplitude values of each note bin for all times 
    map = [list(note_data[time].values()) for time in range(duration)] 
    # flip rows and cols -> pitch class as rows and times as cols 
    chroma_map = pd.DataFrame(map, columns=notes)
    seq = midi_data[midi_seq]
    # check if sequence is in a minor or major scale
    if seq['mode'] == 'm':
        seq_scale = get_minor_scale(notes, midi_seq)
    else:
        seq_scale = get_maj_scale(notes, midi_seq)
    # get current chords from chromagram
    roots_in_seq = list(set(progression))
    # join list of chords into a single string to parse for regex and replace in chroma labels
    # each seq_scale is of length 7
    scale_to_string = (' ').join(seq_scale)
    # get labels for all timepoints 
    chroma_map.insert(12, 'Chord Actual', value=progression)
    # get chord list from the sequence 
    # realign chord labels for minor and diminished chords 
    for chord in roots_in_seq:
        if chord not in seq_scale:
            # define regex pattern to get an instance of the chord 
            pattern = fr'\b\w*{chord}\w*\b'
            # replace the root labels with the true 
            true_chord = re.findall(pattern, scale_to_string, flags=re.IGNORECASE)[0]
            # realign the labels from sequence scale information
            chroma_map['Chord Actual'] = chroma_map['Chord Actual'].replace(to_replace=chord, value=true_chord)
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
def get_chromagram_plot(midi_seq):
    # get map of times and note amplitudes at each time
    note_data = get_note_map(midi_seq)
    # get progression from roots
    progression = get_progression(midi_seq)
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
    # # apply filter to heatmap
    # data_filter = gaussian_filter(chroma_map, sigma=0.5)  # Adjust sigma for more or less blur
    # # overlay blurred data
    # sns.heatmap(data_filter, cmap='viridis', alpha=0.5, cbar=False)
    # make notes from C-Bb instead of Bb-C on plot
    plt.gca().invert_yaxis()
    plt.show()
