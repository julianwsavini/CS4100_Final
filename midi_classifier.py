from sklearn.preprocessing import StandardScaler
import mido
import numpy as np
import pandas as pd
import librosa
import pickle
import warnings
from hmm_classify import predict_next_chords, get_model
warnings.filterwarnings(action='ignore')
import fluidsynth

# Initialize FluidSynth
soundfont = r"undertale.sf2"
synth = fluidsynth.Synth()
synth.start()
# load a soundfont -> accepted file types: .sf2, .sf3
sfid = synth.sfload(soundfont)
# select sound font and initialize synth
synth.program_select(0, sfid, 0, 0)
fluidsynth.init(soundfont)
# load the saved classifier model from the file
vote_classifier = pickle.load(open('voting.pkl', "rb"))
notes = ["C", "C#/Db", "D", "D#", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
# dict to represent active notes and their amplitudes, resets at each note off event
active_notes = {}

"""
use active notes to get input for classifier prediction
    Params: active notes dict
    Outputs: 1 x 12 vector for classifier input
"""
def get_input_for_pred(active_notes):
    input_vect = np.zeros(12)
    for note, velocity in active_notes.items():
        note_idx = notes.index(note)
        input_vect[note_idx] = velocity
    scaled_input = pd.DataFrame(input_vect).transpose()
    scaled_input.columns = notes
    return scaled_input
num_chord = 0
model = get_model()
# initialize an empty chromagram of length 16th to add each change in chord 
curr_chroma = pd.DataFrame(np.zeros(shape = (16, 12)), columns=notes)
curr_chroma['Chord Actual'] = ['' for _ in range(16)]
 
"""
reset to empty chroma for next predictions after HMM prediction
"""
def reset_chroma(curr_chroma):
    curr_chroma = pd.DataFrame(np.zeros(shape = (16, 12)), columns=notes)
    curr_chroma['Chord Actual'] = ['' for _ in range(16)]
    return curr_chroma

 
'''
Add a row to the chromagram given the current state of the chroma,
    Params: current chroma, the number of chords that have been played since the last HMM result, active notes dict, and prediction from voting classifier
    Outputs: new chromagram with row added
'''
def add_to_chromagram(curr_chroma, num_chord, active_notes, prediction):
    note_amps = np.zeros(12)
    new_chroma = curr_chroma.copy()
    for note, velocity in active_notes.items():
        note_idx = notes.index(note)
        note_amps[note_idx] = velocity
    new_chroma.loc[num_chord, curr_chroma.columns[0:12]] = note_amps
    new_chroma.loc[num_chord, 'Chord Actual'] = prediction
    return new_chroma

 
'''
Open midi inport
At all note amplitude/active note changes, predict chord. If 16 chords have been played the HMM will output the first most likely chord
'''
with mido.open_input() as inport:
        try:
            for msg in inport:
                if msg.type == 'note_on':
                    if msg.velocity != 0:
                        # use librosa to get note conversion
                        # use mido to get note and velocity output
                        active_notes[librosa.midi_to_note(msg.note, octave=False, unicode=False)] = msg.velocity
                        for note in active_notes:
                            synth.noteon(0, msg.note, msg.velocity)
                        # if at least 2 notes are playing it is a chord and the classifier will predict at timestep
                        if len(active_notes) > 2:
                            pred_input = get_input_for_pred(active_notes)
                            prediction = vote_classifier.predict(pred_input)[0]
                            print(f'Chord Recognized: {prediction}')
                            # predict hmm
                            if num_chord == 16:
                                curr_chroma = add_to_chromagram(curr_chroma, num_chord, active_notes, prediction)
                                hmm_prediction = predict_next_chords(model, curr_chroma, 2)
                                print(f'HMM Predicted Next Result: {hmm_prediction[0]}')
                                curr_chroma = reset_chroma(curr_chroma)
                                num_chord = 0
                            else:
                                curr_chroma = add_to_chromagram(curr_chroma, num_chord, active_notes, prediction)
                                num_chord += 1
                    else:
                        # reset active notes
                        active_notes = {} 
                        synth.noteoff(0, msg.note)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up before exiting
            inport.close()
            synth.delete()