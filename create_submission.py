import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import os

class SubmissionASR:
    def __init__(self):
        print("Loading ASR model for submission...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
    def transcribe_audio(self, audio_path):
        """Transcribe a single audio file"""
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
            return "transcription not available"
    
    def transcribe_directory(self, input_dir, output_file):
        """Transcribe all audio files in directory for submission"""
        # Get all audio files
        audio_files = []
        for ext in ['.wav', '.flac', '.mp3', '.WAV', '.FLAC', '.MP3']:
            audio_files.extend([f for f in os.listdir(input_dir) if f.endswith(ext)])
        
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        # Transcribe each file
        results = {}
        for audio_file in audio_files:
            audio_path = os.path.join(input_dir, audio_file)
            print(f"Transcribing: {audio_file}")
            transcription = self.transcribe_audio(audio_path)
            results[audio_file] = transcription
            print(f"  -> {transcription[:50]}...")
        
        # Write results
        with open(output_file, 'w', encoding='utf-8') as f:
            for audio_file, transcription in results.items():
                f.write(f"{audio_file} {transcription}\n")
        
        print(f"\n✅ SUBMISSION READY!")
        print(f"Predictions saved to: {output_file}")
        print(f"Total files processed: {len(results)}")

def main():
    asr = SubmissionASR()
    
    # For the challenge, they will provide the input directory
    # You can test with your data first
    test_input_dir = "data/real_distorted"  # Replace with actual test folder
    output_file = "submission_predictions.txt"
    
    if os.path.exists(test_input_dir):
        asr.transcribe_directory(test_input_dir, output_file)
    else:
        print(f"Test directory {test_input_dir} not found")
        print("When you get the test files, run:")
        print(f"python create_submission.py --input_dir [TEST_FOLDER] --output_file {output_file}")

if __name__ == "__main__":
    main()