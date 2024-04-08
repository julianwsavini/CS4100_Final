import joblib
# import pandas as pd
from sklearn.preprocessing import StandardScaler
import mido
import numpy as np
import pandas as pd
import librosa
import pickle 
import warnings
warnings.filterwarnings(action='ignore')

# # Load the saved model from the file
vote_classifier = pickle.load(open('dt.pkl', "rb"))

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
active_notes = {}

def get_input_for_pred(active_notes):
    input_vect = np.zeros(12)
    for note, velocity in active_notes.items():
        note_idx = notes.index(note)
        input_vect[note_idx] = velocity
    scaled_input = pd.DataFrame(input_vect).transpose()
    scaled_input.columns = notes
    return scaled_input

with mido.open_input() as inport:
    for msg in inport:
        if msg.type == 'note_on':
            if msg.velocity != 0:
                active_notes[librosa.midi_to_note(msg.note, octave=False, unicode=False)] = msg.velocity
                if len(active_notes) > 2:
                    pred_input = get_input_for_pred(active_notes)
                    prediction = vote_classifier.predict(pred_input)[0]
                    print(f'Chord Prediction: {prediction}')
            else:
                active_notes = {}