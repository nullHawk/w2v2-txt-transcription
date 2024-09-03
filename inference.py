import torch
from fairseq.data.data_utils import post_process
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
import torchaudio
from omegaconf import OmegaConf
from jiwer import wer, cer
import librosa
import os

def split_audio(audio_path, max_duration=30):
    waveform, sample_rate = torchaudio.load(audio_path)
    total_duration = waveform.size(1) / sample_rate
    
    if total_duration <= max_duration:
        return [(audio_path, waveform)]
    
    chunks = []
    chunk_size = int(max_duration * sample_rate)
    
    for start in range(0, waveform.size(1), chunk_size):
        end = min(start + chunk_size, waveform.size(1))
        chunk_waveform = waveform[:, start:end]
        chunk_path = f"{audio_path}_chunk_{start // chunk_size}.wav"
        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        chunks.append((chunk_path, chunk_waveform))
    
    return chunks

def preprocess_audio(waveform, sample_rate, task):
    assert sample_rate == task.cfg.sample_rate, "Sample rate must match the task sample rate."
    waveform = waveform - waveform.mean()
    waveform = waveform / waveform.abs().max()
    return waveform

def ctc_decode(logits, dictionary):
    tokens = logits.argmax(dim=-1).squeeze().tolist()
    if isinstance(tokens, int):
        tokens = [tokens]
    decoded = []
    prev_token = None
    for token in tokens:
        if token != prev_token and token != dictionary.pad():
            decoded.append(token)
        prev_token = token
    return decoded

def tokens_to_string(tokens, dictionary):
    return post_process(dictionary.string(tokens), symbol='letter')

def transcribe_audio(config_path, checkpoint_path, dictionary_path, audio_path, use_cuda=True):
    config = OmegaConf.load(config_path)
    task = AudioFinetuningTask.setup_task(config.task)
    model, _, _ = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = model[0]
    model.eval()

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    grapheme_dict = Dictionary.load(dictionary_path)
    
    audio_chunks = split_audio(audio_path)
    transcriptions = []

    for chunk_path, waveform in audio_chunks:
        waveform = preprocess_audio(waveform, torchaudio.info(chunk_path).sample_rate, task)
        
        if use_cuda and torch.cuda.is_available():
            waveform = waveform.cuda()

        net_input = {
            'source': waveform,
            'padding_mask': None,
            'src_lengths': torch.tensor([waveform.size(1)])
        }

        if use_cuda and torch.cuda.is_available():
            net_input['src_lengths'] = net_input['src_lengths'].cuda()

        with torch.no_grad():
            net_output = model(**net_input)
            grapheme_lprobs = model.get_normalized_probs(net_output, log_probs=True)

        grapheme_preds = ctc_decode(grapheme_lprobs, grapheme_dict)
        grapheme_transcription = tokens_to_string(grapheme_preds, grapheme_dict)
        transcriptions.append(grapheme_transcription)

        if chunk_path != audio_path:  # Cleanup chunk files
            os.remove(chunk_path)

    final_transcription = " ".join(transcriptions)

    output_dir = "transcripts/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(audio_path)}_transcription.txt")

    with open(output_path, 'w') as f:
        f.write(final_transcription + '\n')

    print(f"Transcription for {audio_path}:")
    print(final_transcription)

    return output_path

def main():
    config_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/finetune_configs/ai4b_xlsr.yaml'
    dictionary_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/dataset/hindi_g/dict.ltr.txt'
    checkpoint_path = '/raid/ganesh/pdadiga/rishabh/asr/IndicWav2Vec/bgpt/models/test_hindi/checkpoint_best.pt'
    audio_path = '/path/to/your/audio/file.wav'

    transcribe_audio(config_path, checkpoint_path, dictionary_path, audio_path, use_cuda=True)

if __name__ == '__main__':
    main()
