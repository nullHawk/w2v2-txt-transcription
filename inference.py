import torch
from fairseq.data.data_utils import post_process
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary
import torchaudio
from omegaconf import OmegaConf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

transcribed_files = set()
model_cache = None

def load_model_and_dict(config_path, checkpoint_path, dictionary_path, use_cuda=True):
    global model_cache
    if model_cache is None:
        config = OmegaConf.load(config_path)
        task = AudioFinetuningTask.setup_task(config.task)
        model, _, _ = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        model = model[0]
        model.eval()
        
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
        
        grapheme_dict = Dictionary.load(dictionary_path)
        model_cache = (model, task, grapheme_dict)
        
    return model_cache

def split_audio(audio_path, max_duration=30, chunks_dir="chunks"):
    os.makedirs(chunks_dir, exist_ok=True)

    waveform, sample_rate = torchaudio.load(audio_path)
    total_duration = waveform.size(1) / sample_rate
    
    if total_duration <= max_duration:
        return [(audio_path, waveform)]
    
    chunks = []
    chunk_size = int(max_duration * sample_rate)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for start in range(0, waveform.size(1), chunk_size):
        end = min(start + chunk_size, waveform.size(1))
        chunk_waveform = waveform[:, start:end]
        chunk_path = os.path.join(chunks_dir, f"{base_name}_chunk_{start // chunk_size}.wav")
        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        chunks.append((chunk_path, chunk_waveform))
    
    return chunks

def preprocess_audio(waveform, sample_rate, task):
    task_sample_rate = getattr(task.cfg, 'sample_rate', 16000)  # Default to 16000 Hz if not in config
    assert sample_rate == task_sample_rate, "Sample rate must match the task sample rate."
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
    start_time = time.time()  # Start time tracking
    
    model, task, grapheme_dict = load_model_and_dict(config_path, checkpoint_path, dictionary_path, use_cuda)
    
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

    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"Transcription result for {audio_path}:")
    print(final_transcription)
    print(f"Transcription for {audio_path} took {elapsed_time:.2f} seconds.")

    return output_path

class AudioHandler(FileSystemEventHandler):
    def __init__(self, config_path, checkpoint_path, dictionary_path, use_cuda=True):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dictionary_path = dictionary_path
        self.use_cuda = use_cuda

    def on_created(self, event):
        if event.is_directory:
            return

        audio_path = event.src_path
        if audio_path in transcribed_files:
            print(f"File {audio_path} already transcribed.")
            return

        print(f"New audio file detected: {audio_path}")
        transcribe_audio(self.config_path, self.checkpoint_path, self.dictionary_path, audio_path, self.use_cuda)
        transcribed_files.add(audio_path)

def main():
    config_path = '/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/config/ai4b_xlsr.yaml'
    dictionary_path = '/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/config/dic.ltr.txt'
    checkpoint_path = '/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/models/checkpoint_best.pt'
    
    # Initialize the handler
    event_handler = AudioHandler(config_path, checkpoint_path, dictionary_path)

    # Set the directory to watch
    audio_directory = '/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/input/'

    # Initialize the observer
    observer = Observer()
    observer.schedule(event_handler, audio_directory, recursive=False)
    load_model_and_dict(config_path, checkpoint_path, dictionary_path, True)
    print(f"Monitoring directory: {audio_directory}")
    
    # Start the observer
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == '__main__':
    main()
