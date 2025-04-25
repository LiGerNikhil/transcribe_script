import argparse
import whisper
import os
import sys
from datetime import datetime
import time
import subprocess
import tempfile
from pydub import AudioSegment

def parse_args():
    """Parse command line arguments with detailed help messages."""
    p = argparse.ArgumentParser(prog="transcribe.py",
        description="Transcribe audio files (MP3/WAV/OGG) using OpenAI Whisper")
    p.add_argument("--model", type=str, default="small",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Model size (small=good balance, medium/large=better accuracy)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to input audio file (supports .mp3, .wav, .ogg)")
    p.add_argument("--output", type=str, default=None,
                   help="Output transcript filename (default: [input]_transcript.txt)")
    p.add_argument("--language", type=str, default=None,
                   help="Force language (e.g. 'en'), auto-detected if omitted")
    p.add_argument("--beam_size", type=int, default=5,
                   help="Beam search size (higher=more accurate but slower)")
    p.add_argument("--no_fp16", action="store_true",
                   help="Disable FP16 (use FP32 instead - better for CPU-only)")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed processing information")
    p.add_argument("--keep_temp", action="store_true",
                   help="Keep temporary converted audio files (for debugging)")
    return p.parse_args()

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S')

def try_direct_processing(input_path, verbose=False):
    """Attempt to process file directly without conversion."""
    if verbose:
        print("Attempting direct processing...")
    return input_path  # Return original path

def convert_with_ffmpeg(input_path, output_path, verbose=False):
    """Convert audio using FFmpeg (best quality)."""
    try:
        if verbose:
            print(f"Converting with FFmpeg to {output_path}...")
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:a', 'libmp3lame', '-q:a', '2',  # Quality setting
            '-y', output_path  # Overwrite if exists
        ]
        subprocess.run(cmd, check=True, capture_output=not verbose)
        return True
    except Exception as e:
        if verbose:
            print(f"FFmpeg conversion failed: {str(e)}")
        return False

def convert_with_pydub(input_path, output_path, verbose=False):
    """Convert audio using pydub (fallback method)."""
    try:
        if verbose:
            print(f"Converting with pydub to {output_path}...")
        
        from pydub import AudioSegment
        if input_path.lower().endswith('.ogg'):
            audio = AudioSegment.from_ogg(input_path)
        else:
            audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        if verbose:
            print(f"Pydub conversion failed: {str(e)}")
        return False

def handle_audio_conversion(input_path, verbose=False, keep_temp=False):
    """
    Automatically handle audio format conversion with multiple fallback methods.
    Returns path to processed audio file.
    """
    # Don't convert if already in supported format
    if input_path.lower().endswith(('.mp3', '.wav', 'm4a')):
        return input_path
        
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=not keep_temp)
    temp_path = temp_file.name
    
    # Try conversion methods in order of preference
    if convert_with_ffmpeg(input_path, temp_path, verbose):
        return temp_path
    if convert_with_pydub(input_path, temp_path, verbose):
        return temp_path
        
    # Final fallback - try direct processing
    return try_direct_processing(input_path, verbose)

def main():
    args = parse_args()
    
    # 1. Validate input file
    if not os.path.isfile(args.input):
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # 2. Handle audio conversion if needed
    original_input = args.input
    temp_audio = None
    
    try:
        if args.verbose:
            print(f"Processing input file: {original_input}")
        
        args.input = handle_audio_conversion(
            args.input,
            verbose=args.verbose,
            keep_temp=args.keep_temp
        )
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {str(e)}", file=sys.stderr)
        print("Possible solutions:", file=sys.stderr)
        print("1. Install FFmpeg (https://ffmpeg.org)", file=sys.stderr)
        print("2. pip install pydub", file=sys.stderr)
        print("3. Convert file manually to WAV/MP3", file=sys.stderr)
        sys.exit(1)

    # 3. Set output filename
    if args.output is None:
        base_name = os.path.splitext(original_input)[0]
        args.output = f"{base_name}_transcript.txt"

    # 4. Load Whisper model
    try:
        if args.verbose:
            print(f"Loading {args.model} model...")
        
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"[ERROR] Model loading failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # 5. Run transcription
    try:
        if args.verbose:
            print("Starting transcription...")
        
        result = model.transcribe(
            args.input,
            language=args.language,
            beam_size=args.beam_size,
            fp16=not args.no_fp16
        )
    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # 6. Save results
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            # Write metadata header
            f.write(f"Transcript for: {os.path.basename(original_input)}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {args.model}\n")
            if args.input != original_input:
                f.write(f"Note: Audio was converted for processing\n")
            f.write("="*50 + "\n\n")
            
            # Write each segment with timestamps
            for seg in result["segments"]:
                f.write(f"[{format_time(seg['start'])} - {format_time(seg['end'])}] "
                       f"{seg['text'].strip()}\n\n")
        
        print(f"Success! Transcript saved to: {args.output}")
    except Exception as e:
        print(f"[ERROR] Failed to save transcript: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()