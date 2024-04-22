from utils import NOTES_NAMES, FULL_CHORD_LIST, CUSTOM_ENCODING, INVERSE_ENCODING, separate_last_chord, mean_chord_distance_with_quality, predict_next_chords, calculate_covariance_from_chroma, separate_for_training, calculate_mu_from_chroma, calculate_transition_probabilites, format_indiv_chroma, get_unique_predicted, calculate_initial_probabilities
import pickle
from chroma import get_chromagram
import pandas as pd
from tqdm import tqdm
import numpy as np
from hmmlearn import hmm


# Load data
with open(r"dataset.pkl", 'rb') as data:
    midi_data:dict = pickle.load(data)

def get_model():
    """Split the data into testing and training, make a GaussianHMM"""
    training_piece_names, test_piece_names = separate_for_training(midi_data, 0.8)

    song_chromagrams = []
    for song_name in tqdm(list(training_piece_names)):
        indiv_chroma = get_chromagram(song_name, midi_data)
        formatted = format_indiv_chroma(indiv_chroma)
        song_chromagrams.append(indiv_chroma)

    chromagram = pd.concat(song_chromagrams)
    chromagram.head(200)

    initial_state_probabilties = calculate_initial_probabilities(training_piece_names, midi_data)
    transition_prob_matrix = calculate_transition_probabilites(chromagram)
    mu = calculate_mu_from_chroma(chromagram)
    covars = calculate_covariance_from_chroma(chromagram)

    model = hmm.GaussianHMM(n_components=transition_prob_matrix.shape[0], covariance_type="diag")
    model.startprob_ = initial_state_probabilties
    model.transmat_ = transition_prob_matrix.values
    model.means_ = mu
    model.covars_ = np.array([np.diag(cov_matrix) + 1e-6 for cov_matrix in covars]).reshape(-1, 12)
    model.n_features = 36
    return model


