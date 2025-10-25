import os

def check_data_structure():
    print("Checking data structure...")
    
    # Check distorted folder
    distorted_path = "data/distorted"
    if os.path.exists(distorted_path):
        files = os.listdir(distorted_path)
        print(f"Files in distorted folder: {len(files)}")
        for file in files[:10]:  # Show first 10 files
            print(f"  - {file}")
    else:
        print("Distorted folder does not exist!")
    
    # Check if transcripts.txt exists
    transcript_file = "data/distorted/transcripts.txt"
    if os.path.exists(transcript_file):
        print(f"\nTranscript file exists!")
        # Show first few lines
        with open(transcript_file, 'r') as f:
            lines = f.readlines()[:5]
            print("First 5 lines:")
            for line in lines:
                print(f"  - {line.strip()}")
    else:
        print(f"\nTranscript file NOT found at: {transcript_file}")
    
    # Check clean data
    clean_path = "data/clean"
    if os.path.exists(clean_path):
        clean_files = []
        for root, dirs, files in os.walk(clean_path):
            for file in files:
                if file.endswith('.flac'):
                    clean_files.append(os.path.join(root, file))
        print(f"\nFound {len(clean_files)} clean .flac files")
    else:
        print("Clean folder does not exist!")

if __name__ == "__main__":
    check_data_structure()