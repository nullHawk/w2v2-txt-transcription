# W2V2-txt-interference-pipeline

This repository contains a Python script that transcribes audio files using a w2v2 pretrained model. The script splits the audio into chunks, processes each chunk, and then decodes the transcriptions using a CTC-based model. The final transcription is saved as a text file.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#script-overview)
- [Arguments](#arguments)
- [Output](#output)
- [License](#license)

## Requirements

To run this script, you need the following Python packages:

- `torch`
- `fairseq`
- `torchaudio`
- `omegaconf`
- `jiwer`
- `librosa`

You can install these packages using pip:

```bash
pip install torch fairseq torchaudio omegaconf jiwer librosa
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/nullHawk/w2v2-txt-transcription.git
   cd w2v2-txt-transcription
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary model checkpoints and configuration files. Modify the paths in the script or command accordingly.

## Usage

To transcribe an audio file, run the following command:

```bash
python transcribe.py /path/to/your/audio/file.mp3
```

If no audio file path is provided, the script will use a default path defined in the `main()` function.

## Script Overview

### `split_audio(audio_path, max_duration=30)`

Splits the audio file into chunks if it exceeds the `max_duration` in seconds.

### `preprocess_audio(waveform, sample_rate, task)`

Normalizes the audio waveform and ensures the sample rate matches the task's requirements.

### `ctc_decode(logits, dictionary)`

Decodes the model's output logits to token sequences using the CTC algorithm.

### `tokens_to_string(tokens, dictionary)`

Converts the token sequences into human-readable strings.

### `transcribe_audio(config_path, checkpoint_path, dictionary_path, audio_path, use_cuda=True)`

Main function that handles the loading of the model, processing of the audio, and transcription.

### `main()`

Handles command-line arguments and initiates the transcription process.

## Arguments

- **config_path**: Path to the model configuration file (`.yaml`).
- **checkpoint_path**: Path to the pretrained model checkpoint file (`.pt`).
- **dictionary_path**: Path to the dictionary file (`dic.ltr.txt`).
- **audio_path**: Path to the audio file to be transcribed.
- **use_cuda**: (Optional) Set to `True` to use GPU for transcription, defaults to `True`.

## Output

The script generates a transcription text file for the provided audio file and saves it in the `transcripts/` directory.

Example output file name:

```
transcripts/audio_file_name_transcription.txt
```

The console will also display the transcription and the time taken for the process.

