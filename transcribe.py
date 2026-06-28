import sys
import os
import torch
import torchaudio
from transformers import AutoModel

def main():
    audio_file = "sample.wav"
    language = "hi"

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    if len(sys.argv) > 2:
        language = sys.argv[2]
        
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return

    print("Loading IndicConformer model... (this may take a while if downloading for the first time)")
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

    print(f"Transcribing '{audio_file}' in language '{language}'...")
    
    wav, sr = torchaudio.load(audio_file)
    wav = torch.mean(wav, dim=0, keepdim=True)
    target_sample_rate = 16000
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
        
    transcription_rnnt = model(wav, language, "rnnt")
    text = transcription_rnnt[0] if isinstance(transcription_rnnt, list) else str(transcription_rnnt)

    print("\n--- Transcription Result ---")
    print(text)
    print("----------------------------\n")

if __name__ == "__main__":
    main()
