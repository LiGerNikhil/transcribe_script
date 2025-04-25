import argparse
import whisper
import os
import sys
from datetime import datetime
import torch
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Optimized Whisper transcription")
    p.add_argument("--input", required=True, help="Input audio file path")
    p.add_argument("--output", default=None, help="Output text file path")
    p.add_argument("--model", default="base", choices=["tiny", "base", "small"],
                  help="Model size (base recommended for speed/accuracy balance)")
    p.add_argument("--language", default=None, help="Language code (e.g. 'en')")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                  help="Device to use (cuda/cpu/mps)")
    p.add_argument("--fp16", action="store_true", help="Use FP16 precision (faster but requires GPU)")
    return p.parse_args()

def load_model(model_size, device, use_fp16):
    """Load and optimize the Whisper model"""
    model = whisper.load_model(model_size, device=device)
    if device == "cuda" and use_fp16:
        model = model.half()  # Convert to FP16 for faster processing
    return model

def transcribe_fast(model, audio_path, language=None, use_fp16=False):
    """Optimized transcription function with progress tracking"""
    options = {
        "language": language,
        "fp16": use_fp16,
    }
    options = {k: v for k, v in options.items() if v is not None}
    
    # Initialize progress bar
    pbar = tqdm(total=100, desc="Transcribing")
    
    # Custom callback for progress updates
    def progress_callback(current, total):
        percent = (current / total) * 100
        pbar.update(percent - pbar.n)
    
    try:
        result = model.transcribe(audio_path, **options)
        pbar.close()
        return result
    except Exception as e:
        pbar.close()
        raise e

def main():
    args = parse_args()
    
    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: File not found - {args.input}", file=sys.stderr)
        sys.exit(1)
        
    # Set output path
    output_path = args.output or f"{os.path.splitext(args.input)[0]}_transcript.txt"
    
    # Check device support
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = "cpu"
        args.fp16 = False
    
    # Load model
    try:
        print(f"Loading {args.model} model on {args.device}...")
        model = load_model(args.model, args.device, args.fp16)
    except Exception as e:
        print(f"Model loading failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Transcribe
    try:
        result = transcribe_fast(model, args.input, args.language, args.fp16)
    except Exception as e:
        print(f"Transcription failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Save results
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {os.path.basename(args.input)}\n")
            f.write(f"Model: {args.model} | Device: {args.device} | FP16: {args.fp16}\n")
            f.write("="*50 + "\n\n")
            
            for seg in result["segments"]:
                start = datetime.utcfromtimestamp(seg['start']).strftime('%H:%M:%S')
                end = datetime.utcfromtimestamp(seg['end']).strftime('%H:%M:%S')
                f.write(f"[{start} - {end}] {seg['text'].strip()}\n\n")
        
        print(f"\nTranscript saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save output: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()