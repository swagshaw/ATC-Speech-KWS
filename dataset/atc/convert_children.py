from audiomentations import Compose, PitchShift, TimeStretch
import numpy as np
import soundfile as sf
import os
def process_audio_to_adult_speech(input_path, output_path):
    """
    Apply transformations to an audio file to make children's speech sound more like adult speech.

    Parameters:
    - input_path: Path to the input audio file (children's speech).
    - output_path: Path where the processed audio file will be saved.
    """
    # Load the audio file
    audio, sample_rate = sf.read(input_path)

    # Define the augmentation pipeline
    augment = Compose([
        # Shift pitch down. The factor might need adjustment.
        PitchShift(min_semitones=-6.5, max_semitones=-3.5, p=1.0),

        # Stretch time slightly. A value less than 1 slows down.
        TimeStretch(min_rate=0.9, max_rate=1.0, p=0.5),
    ])

    # Apply augmentations
    processed_audio = augment(samples=audio, sample_rate=sample_rate)
    processed_audio = np.asarray(processed_audio, dtype=np.float64)
    # Save the processed audio to a file
    sf.write(output_path, processed_audio, sample_rate)

def convert_and_move_dataset(unprocessed_data_folder, processed_data_folder):
    print(f"Converting and moving dataset from {unprocessed_data_folder} to {processed_data_folder}...")
    for root, dirs, files in os.walk(unprocessed_data_folder):
        print(f"Processing folder: {root}")
        for file in files:
            print(f"Processing: {file}")
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                
                # Construct the new path in the "/data" folder, preserving the class_label subfolder structure
                output_path = input_path.replace(unprocessed_data_folder, processed_data_folder)
                
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process and save the audio file
                process_audio_to_adult_speech(input_path, output_path)
                print(f"Processed and moved: {output_path}")



unprocessed_data_folder = "/home/yangxiao/TorchKWS/dataset/atc/un_processed_data"
processed_data_folder = "/home/yangxiao/TorchKWS/dataset/atc/data"
convert_and_move_dataset(unprocessed_data_folder, processed_data_folder)