import pickle
import librosa as lbs
from pychord import analyzer
import re

with open(r"C:\Users\Gianna\Downloads\dataset.pkl", 'rb') as data:
    midi_data = pickle.load(data)
## NEXT STEPS:
    # use velocity to determine amplitude -> create FFT and create Pitch Class profile, then do expectation maximization/vertibi algorithm 


    # construct a distribution based on amplitude (velocity) at each timepoint to create a Fast fourier transform, which represents sampling at each time point,
    # where time is on x axis and frequency/amplitude is on the y axis 
    # we make this by getting the frequency at a time point (highest MIDI value, value 3 in the note matrix) and sampling at a rate of 2x the max frequency



    # then use the log math equation to construct a PCP matrix, which is a 12 d matrix with notes C-Bb as rows and colums
    # each value within the 12 d matrix is an amplitude/intensity, which corresponds to the velocity at each beat (last value in the note matrix)

    # from the PCP matrix we get an HMM from expectation maximization during training 


# create dictionary for progression
# where each key is the # eigth note in the sequence 
# the values are a list of notes/pitches that are playing at that note 

def get_note_map(nmat):
    # get end of sequence from last row 
    len_sequence = nmat[-1][1]
    beats = range(len_sequence + 1)
    # create empty map with the # of beats as the key
    note_map = {key:[] for key in beats}
    for row in nmat:
        start = row[0]
        end = row[1]
        note = lbs.midi_to_note(row[2], unicode=False)
        for beat in range(start, end + 1):
            note_map[beat].append(note)
    return note_map


# given the note matrix which represents notes playing at each beat, get the chord dictionary at each beat
# keys are beat # and 
def chord_map_from_note(note_map):
    # use regex to remove octave info from note map
    pattern = re.compile(r'\d')
    
    note_map_no_octave = {beat:list(set(map((lambda x: re.sub(pattern, '', x)), notes))) for beat, notes in note_map.items()}
    print(note_map_no_octave)
    chords_map = {key:[] for key in range(len(note_map_no_octave))}

    for beat, notes in note_map_no_octave.items():
        if len(notes) != 0:
            
            chord = analyzer.find_chords_from_notes(notes)
            chords_map[beat].append(chord)
    return chords_map