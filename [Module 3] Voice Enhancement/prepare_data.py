import os
import soundfile as sf
from data_tools import audio_files_to_numpy
from data_tools import blend_noise_randomly, numpy_audio_to_matrix_spectrogram
import numpy as np

def create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
               min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise, and clean voices to disk as well as complex phase,
    time series, and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in the args module"""

    print("STARTING DATA PREPARATION")

    # List the noise and voice audio files
    print("Listing noise and voice files...")
    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)
    print(f"Initial number of noise files: {len(list_noise_files)}")
    print(f"Initial number of voice files: {len(list_voice_files)}")

    def remove_ds_store(lst):
        """Remove macOS-specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')
            print("Removed .DS_Store file from the list.")
        return lst

    # Remove .DS_Store file if it exists
    print("Removing .DS_Store file if it exists...")
    list_noise_files = remove_ds_store(list_noise_files)
    list_voice_files = remove_ds_store(list_voice_files)
    print(f"Number of noise files after removal: {len(list_noise_files)}")
    print(f"Number of voice files after removal: {len(list_voice_files)}")

    nb_voice_files = len(list_voice_files)
    nb_noise_files = len(list_noise_files)

    # Extract noise and voice files from directories and convert them to NumPy arrays
    print("Converting noise files to NumPy arrays...")
    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                                 frame_length, hop_length_frame_noise, min_duration)
    print(f"Noise data converted: shape = {noise.shape}")

    print("Converting voice files to NumPy arrays...")
    voice = audio_files_to_numpy(voice_dir, list_voice_files,
                                 sample_rate, frame_length, hop_length_frame, min_duration)
    print(f"Voice data converted: shape = {voice.shape}")

    # Blend clean voice with random noise
    print("Blending clean voice with random noise...")
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
        voice, noise, nb_samples, frame_length)
    print("Voice and noise blending completed.")
    print(f"prod_voice shape: {prod_voice.shape}")
    print(f"prod_noise shape: {prod_noise.shape}")
    print(f"prod_noisy_voice shape: {prod_noisy_voice.shape}")

    # Save generated long audio files to disk for quality check
    print("Saving long audio files for quality check...")
    noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
    sf.write(os.path.join(path_save_sound, 'noisy_voice_long.wav'), noisy_voice_long[0, :], sample_rate)
    print(f"Saved: {os.path.join(path_save_sound, 'noisy_voice_long.wav')}")

    voice_long = prod_voice.reshape(1, nb_samples * frame_length)
    sf.write(os.path.join(path_save_sound, 'voice_long.wav'), voice_long[0, :], sample_rate)
    print(f"Saved: {os.path.join(path_save_sound, 'voice_long.wav')}")

    noise_long = prod_noise.reshape(1, nb_samples * frame_length)
    sf.write(os.path.join(path_save_sound, 'noise_long.wav'), noise_long[0, :], sample_rate)
    print(f"Saved: {os.path.join(path_save_sound, 'noise_long.wav')}")

    # Determine the spectrogram size
    print("Determining spectrogram size...")
    dim_square_spec = int(n_fft / 2) + 1
    print(f"Spectrogram size: {dim_square_spec}")

    # Generate amplitude and phase spectrograms for clean voice
    print("Generating amplitude and phase spectrograms for clean voice...")
    m_amp_db_voice, m_pha_voice = numpy_audio_to_matrix_spectrogram(
        prod_voice, dim_square_spec, n_fft, hop_length_fft)
    print(f"Clean voice spectrogram created: amplitude shape = {m_amp_db_voice.shape}, phase shape = {m_pha_voice.shape}")

    print("Generating amplitude and phase spectrograms for noise...")
    m_amp_db_noise, m_pha_noise = numpy_audio_to_matrix_spectrogram(
        prod_noise, dim_square_spec, n_fft, hop_length_fft)
    print(f"Noise spectrogram created: amplitude shape = {m_amp_db_noise.shape}, phase shape = {m_pha_noise.shape}")

    print("Generating amplitude and phase spectrograms for noisy voice...")
    m_amp_db_noisy_voice, m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
        prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)
    print(f"Noisy voice spectrogram created: amplitude shape = {m_amp_db_noisy_voice.shape}, phase shape = {m_pha_noisy_voice.shape}")

    # Save time series data to disk for training and quality check
    print("Saving time series data to disk...")
    np.save(os.path.join(path_save_time_serie, 'voice_timeserie.npy'), prod_voice)
    print(f"Saved: {os.path.join(path_save_time_serie, 'voice_timeserie.npy')}")

    np.save(os.path.join(path_save_time_serie, 'noise_timeserie.npy'), prod_noise)
    print(f"Saved: {os.path.join(path_save_time_serie, 'noise_timeserie.npy')}")

    np.save(os.path.join(path_save_time_serie, 'noisy_voice_timeserie.npy'), prod_noisy_voice)
    print(f"Saved: {os.path.join(path_save_time_serie, 'noisy_voice_timeserie.npy')}")

    print("Saving amplitude spectrogram data to disk...")
    np.save(os.path.join(path_save_spectrogram, 'voice_amp_db.npy'), m_amp_db_voice)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'voice_amp_db.npy')}")

    np.save(os.path.join(path_save_spectrogram, 'noise_amp_db.npy'), m_amp_db_noise)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'noise_amp_db.npy')}")

    np.save(os.path.join(path_save_spectrogram, 'noisy_voice_amp_db.npy'), m_amp_db_noisy_voice)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'noisy_voice_amp_db.npy')}")

    print("Saving phase spectrogram data to disk...")
    np.save(os.path.join(path_save_spectrogram, 'voice_pha_db.npy'), m_pha_voice)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'voice_pha_db.npy')}")

    np.save(os.path.join(path_save_spectrogram, 'noise_pha_db.npy'), m_pha_noise)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'noise_pha_db.npy')}")

    np.save(os.path.join(path_save_spectrogram, 'noisy_voice_pha_db.npy'), m_pha_noisy_voice)
    print(f"Saved: {os.path.join(path_save_spectrogram, 'noisy_voice_pha_db.npy')}")

    print("DATA PREPARATION COMPLETED")
