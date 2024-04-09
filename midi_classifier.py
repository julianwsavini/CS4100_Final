from sklearn.preprocessing import StandardScaler
import mido
import numpy as np
import pandas as pd
import librosa
import pickle 
import warnings
warnings.filterwarnings(action='ignore')
import fluidsynth

# Initialize FluidSynth
synth = fluidsynth.Synth()
synth.start()

# Load a soundfont
sfid = synth.sfload(r"Yamaha_MA2.sf2")
synth.program_select(0, sfid, 0, 0)
fluidsynth.init("Yamaha_MA2.sf2")
# # Load the saved model from the file
vote_classifier = pickle.load(open('voting.pkl', "rb"))

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
        try: 
            for msg in inport:
                if msg.type == 'note_on':
                    if msg.velocity != 0:
                        active_notes[librosa.midi_to_note(msg.note, octave=False, unicode=False)] = msg.velocity
                        for note in active_notes:
                            pass
                            #fluidsynth.play_Note(Note(note), msg.velocity)
                        if len(active_notes) > 2:
                            pred_input = get_input_for_pred(active_notes)
                            prediction = vote_classifier.predict(pred_input)[0]
                            print(f'Chord Prediction: {prediction}')
                    else:
                        active_notes = {}  
                        synth.noteoff(0, msg.note)
        except KeyboardInterrupt:
            pass
        finally:
            # Clean up before exiting
            inport.close()
            synth.delete()