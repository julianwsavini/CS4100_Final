import mido
import pygame
import numpy as np

# Initialize Pygame mixer
pygame.mixer.init()

# Define audio parameters
sample_rate = 44100  # Sample rate (samples per second)
duration = 0.5  # Duration of the generated sound in seconds

# Create an empty stereo sound array (silence) using NumPy
silence_samples = np.zeros((int(sample_rate * duration), 2), dtype=np.int16)

# Create a Pygame sound object with silence
silence_sound = pygame.sndarray.make_sound(silence_samples)

# Define a low-pass filter function
def low_pass_filter(x, alpha):
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

# Define a callback function to handle incoming MIDI messages
def on_message(message):
    if message.type == 'note_on':
        # Calculate the frequency of the note
        frequency = 440 * 2 ** ((message.note - 69) / 12)

        # Create a new stereo Pygame sound object for the note (sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time array
        sine_wave = (32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

        # Apply the low-pass filter to the sine wave
        filtered_wave = low_pass_filter(sine_wave, alpha=0.9)

        # Combine left and right channels for stereo
        note_samples = np.column_stack((filtered_wave, filtered_wave))

        # Ensure note_samples is 2D
        note_samples = np.atleast_2d(note_samples)

        # Create a Pygame sound object for the note with stereo samples
        note_sound = pygame.sndarray.make_sound(note_samples)

        # Play the note sound
        note_sound.play()

# Create a MIDI input port
with mido.open_input() as port:
    print("Listening for MIDI messages...")

    # Set the callback function for incoming messages
    for message in port:
        on_message(message)
