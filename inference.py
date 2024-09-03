import torch
from fairseq.data.data_utils import post_process
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
import torchaudio
from omegaconf import OmegaConf
import os

def transcribe_audio(config_path, checkpoint_path, dictionary_path, audio_path, use_cuda=True):
    # Load the configuration
    config = OmegaConf.load(config_path)

    # Initialize the task
    task = AudioFinetuningTask.setup_task(config.task)

    # Load the model checkpoint
    model, cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = model[0]
    model.eval()

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    # Load the dictionary
    grapheme_dict = Dictionary.load(dictionary_path)
    print(f"Number of graphemes in the dictionary: {len(grapheme_dict)}")
    print(f"Grapheme dictionary: {grapheme_dict.symbols}")

    # Function to load and preprocess audio file
    def preprocess_audio(audio_path, task):
        waveform, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == task.cfg.sample_rate, "Sample rate must match the task sample rate."
        # Normalize waveform
        waveform = waveform - waveform.mean()
        waveform = waveform / waveform.abs().max()
        return waveform

    # Function to decode CTC outputs
    def ctc_decode(logits, dictionary):
        tokens = logits.argmax(dim=-1).squeeze().tolist()
        # Collapse repeated tokens and remove blanks (represented by 0)
        if isinstance(tokens, int):
            tokens = [tokens]
        decoded = []
        prev_token = None
        for token in tokens:
            if token != prev_token and token != dictionary.pad():
                decoded.append(token)
            prev_token = token
        return decoded

    # Function to convert token indices to strings
    def tokens_to_string(tokens, dictionary):
        return post_process(dictionary.string(tokens), symbol='letter')

    # Preprocess the audio file
    waveform = preprocess_audio(audio_path, task)

    if use_cuda and torch.cuda.is_available():
        waveform = waveform.cuda()

    # Prepare input sample
    net_input = {
        'source': waveform,
        'padding_mask': None,
    }

    # Add input lengths for inference
    net_input['src_lengths'] = torch.tensor([waveform.size(1)])

    if use_cuda and torch.cuda.is_available():
        net_input['src_lengths'] = net_input['src_lengths'].cuda()

    # Perform inference
    with torch.no_grad():
        net_output = model(**net_input)
        grapheme_lprobs = model.get_normalized_probs(net_output, log_probs=True)

    # Get the predicted tokens for graphemes
    grapheme_preds = ctc_decode(grapheme_lprobs, grapheme_dict)
    grapheme_transcription = tokens_to_string(grapheme_preds, grapheme_dict)

    # Save the transcription to a file
    os.makedirs(f'inference_result/our_dataset', exist_ok=True)
    output_prefix = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = f'inference_result/our_dataset/{output_prefix}_transcription.txt'
    with open(output_path, 'w') as f:
        f.write(grapheme_transcription + '\n')

    print(f"Transcription saved to: {output_path}")

    return output_path

def main():
    # Define input paths and parameters
    config_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/finetune_configs/ai4b_xlsr.yaml'
    dictionary_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/dataset/hindi_g/dict.ltr.txt'
    checkpoint_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/bgpt/models/test_hindi/checkpoint_best.pt'
    audio_path = '/path/to/your/audio/file.mp3'  # Update this path with your actual audio file

    # Perform transcription
    transcribe_audio(config_path, checkpoint_path, dictionary_path, audio_path, use_cuda=True)

if __name__ == '__main__':
    main()
