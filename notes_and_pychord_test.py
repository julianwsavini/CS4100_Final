import pickle
# library to easily access info about chords such as root of chords, a chord given a list of notes, etc
# download with pip install pychord
import pychord
from pychord import analyzer 

with open(r"dataset.pkl", 'rb') as data:
    midi_data = pickle.load(data)

#print(midi_data['Niko_Kotoulas_Plucks_1_G#m-F#-E-C# (vi-V-IV-II).mid'])

## Notes - 
    # Tonic is the octave of the first root note on the scale -> essentially the 'center' of a key 
    # Mode -> either major or minor

    #Note Matrix
    # assumes every note is assumed to be 1/8
    # each row represents the start eigth note in the entire piece, end note
    # pitch (refering to the 10 octaves from C_1 (C in octave 0) to G_9 (G in octave 8))-> 0-127
    # velocity (how hard the midi is being struck) -> usually 0-127

    # Root chord: always the root note of chord represented numerically (Ex. C#m root is C, D root is D, Gm root is G)
    # root and mode can be used together to infer the key -> ex. if a mode is minor and the root is D, then the chord is Dm, 
                                                            #    if a mode is major and the root is G, then the chord is G
    

# list of all notes in order of their numerical value (all scales have 7 total chords)
notes = ['C', 'C#/Db' 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

# In different major and minor scales, next chords are predicted from the 'strength' or 'weakness' of chord, meaning how well 2 chords clash/go together
# Often in music, strong chords are followed by weak chords to resolve tension.
# these strong/weak chords are different for each scale 
# the tonic note of a progression is the root of the first chord of a scale 


# MAJOR SCALES:
# I (main key of scale), II, III, IV, V, VI, VII
# Formula: Major chords = I, III, V (1,3,5); Minor chords = II, IV, VI (2,4,6); Diminished = VII (7, less common besides in jazz music ususally)
# Each chord is either a half step or whole step to the next
# for example the key of C has a whole step to D, which is the second chord of the scale


# Chord 1 (root/tonic chord)-> Major, Strong chord, Provides stability and resolution and is usually the beginning of a chord progression

# Chord 2 (super tonic) -> Minor, Weak chord, Whole step from chord 1

# Chord 3 (mediant) -> Major, Weak chord, Whole step from chord 2

# Chord 4 (subdominant) -> Minor, Strong chord, Half step from 3

# Chord 5 (dominant) -> Major, Strong chord, Half step from 4,  usually strongest chord besides Chord 1, often builds tension to resolve back to root

# Chord 6 (submediant ) -> Minor, Weak Chord, Whole step from 5

# Chord 7 (dim) -> Dim, creates strong tension to often resolve back to root, half step from 1


# MINOR SCALES
# I (main key of scale), II, III, IV, V, VI, VII
# Formula: Minor chords = I, IV, V (1,4,5); Major chords = II, III, VI (2,3,6); Diminished = II 

# Each chord is either a half step or whole step to the next
# for example the key of Cm has a half step to C#dim/Db, which is the second chord of the scale
# if in some scale F# was the 3rd chord, half step up and minor would be Gm
# the only chords that only have a half step between them is E-F and B-C (therefore no such chords as E#/Fb or B#/Cb)


# Chord 1 (root/tonic chord)-> Minor, Strong chord, Root 

# Chord 2 (diminished) -> diminished, builds strong tension (so prob more likely to have high velocity?), half step from 1

# Chord 3 -> Major, Weak , Whole step from 2

# Chord 4 -> Minor, Strong Chord, half step from 3

# Chord 5 -> Minor, Strong, whole step from 4

# Chord 6 -> Major, Strong, half step from 5

# Chord 7 (dominant) -> Major, Strong, Whole step from 6, half step from 1


# function to find chords given the notes that comprise it 
# expected = B
print(analyzer.find_chords_from_notes(['B', 'D#', 'F#']))
# get numeric value of note
print(analyzer.note_to_val('D#'))
print(analyzer.note_to_val('A#'))
# get all possible orderings of these notes 
print(analyzer.get_all_rotated_notes(['B', 'D#', 'F#']))
# libary has other features such as getting root notes from chords and getting additional chord info, ability to transpose etc (move the value of chord either up or down a certain amount of steps on the scale)