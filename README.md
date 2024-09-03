# W2V2-txt-inference-pipeline

This repository contains a Python script that transcribes audio files using a w2v2 pretrained model. The script splits the audio into chunks, processes each chunk, and then decodes the transcriptions using a CTC-based model. The final transcription is saved as a text file.

## Features

- **Automatic Audio Chunking**: Splits audio files into chunks for more efficient transcription.
- **Model Caching**: Loads the model and dictionary once at startup to reduce latency.
- **Real-Time Directory Monitoring**: Uses `watchdog` to monitor a directory and transcribe new audio files as they are added.
- **Time Tracking**: Measures and prints the time taken for each transcription.

## Requirements

- Python 3.7+
- PyTorch
- Fairseq
- torchaudio
- omegaconf
- watchdog

## Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/nullHawk/w2v2-txt-transcription.git
    cd w2v2-txt-transcription
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install torch fairseq torchaudio omegaconf watchdog
    ```

4. **Prepare Configuration Files**:
    - Place your Fairseq configuration file (`ai4b_xlsr.yaml`), dictionary file (`dic.ltr.txt`), and model checkpoint file (`checkpoint_best.pt`) in the appropriate paths.
    - Update the paths in the script to match your file locations.

## Usage

1. **Update the Script**:
    - Ensure the paths for `config_path`, `dictionary_path`, and `checkpoint_path` in the `main()` function of the `transcription.py` script are correctly set.

2. **Run the Script**:
    ```bash
    python transcription.py
    ```

3. **Add Audio Files**:
    - Place audio files in the directory specified by `audio_directory` in the script. The script will automatically detect new files and process them.

## Configuration

- **`config_path`**: Path to the Fairseq configuration file.
- **`dictionary_path`**: Path to the Fairseq dictionary file.
- **`checkpoint_path`**: Path to the pre-trained model checkpoint file.
- **`audio_directory`**: Directory to monitor for new audio files.

