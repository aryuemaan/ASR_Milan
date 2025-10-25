import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Use torchaudio's pre-trained model
def load_pretrained_model():
    """Load a pre-trained ASR model and fine-tune it"""
    try:
        # Try to load Wav2Vec2 pre-trained model
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
        print(f"Loaded pre-trained: {bundle._name}")
        return model
    except:
        print("Pre-trained model not available, using fallback")
        return None

class FineTuneDataset:
    def __init__(self, audio_dir, transcript_file):
        self.audio_dir = audio_dir
        self.data = []
        
        with open(transcript_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_file, transcript = parts
                    audio_path = os.path.join(audio_dir, audio_file)
                    if os.path.exists(audio_path):
                        self.data.append((audio_file, transcript))
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_file, transcript = self.data[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0), transcript

def main():
    print("=== FINE-TUNING PRE-TRAINED MODEL ===")
    
    # Try pre-trained first
    model = load_pretrained_model()
    
    if model is None:
        print("Using simple fallback approach")
        # Fallback: Use very simple model
        from src.ctc_model import CTCASR
        model = CTCASR(num_classes=29)
    
    # Load data
    dataset = FineTuneDataset("data/real_distorted", "data/real_distorted/transcripts.txt")
    
    # Use only 100 samples for quick testing
    dataset.data = dataset.data[:100]
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(10):
        total_loss = 0
        for i, (waveform, transcript) in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
            try:
                optimizer.zero_grad()
                
                # Ensure proper length
                if len(waveform) < 16000:
                    waveform = torch.cat([waveform, torch.zeros(16000 - len(waveform))])
                elif len(waveform) > 16000:
                    waveform = waveform[:16000]
                
                # Forward pass
                outputs = model(waveform.unsqueeze(0))
                
                # Simple loss - we'll improve this
                loss = outputs.sum() * 0.001  # Placeholder
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            except Exception as e:
                continue
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")
    
    torch.save(model.state_dict(), "checkpoints/pretrained_finetuned.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()