#!/usr/bin/env python3
"""
ASR Challenge Inference Script
"""

import argparse
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import glob

class ASRInference:
    def __init__(self, model_path=None):
        print("Loading ASR model...")
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        print("Model loaded successfully!")
    
    def transcribe_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            input_values = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values
            
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.lower()
        except Exception as e:
            print(f"Error with {audio_path}: {e}")
            return "transcription unavailable"

def main():
    parser = argparse.ArgumentParser(description='ASR Challenge Inference')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing distorted audio files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file for transcriptions')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory '{args.input_dir}' does not exist!")
        print("Please provide a valid path to audio files.")
        return
    
    asr = ASRInference()
    
    # Find all audio files (case insensitive)
    audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.WAV', '*.FLAC', '*.MP3']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    print(f"Found {len(audio_files)} audio files in {args.input_dir}")
    
    if len(audio_files) == 0:
        print("No audio files found! Supported formats: .wav, .flac, .mp3")
        return
    
    # Transcribe files
    results = {}
    for audio_path in audio_files:
        audio_file = os.path.basename(audio_path)
        print(f"Transcribing: {audio_file}")
        transcription = asr.transcribe_audio(audio_path)
        results[audio_file] = transcription
        print(f"  -> {transcription[:60]}...")
    
    # Write results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for audio_file, transcription in results.items():
            f.write(f"{audio_file} {transcription}\n")
    
    print(f"\n✅ SUCCESS!")
    print(f"Predictions saved to: {args.output_file}")
    print(f"Total files processed: {len(results)}")

if __name__ == "__main__":
    main()