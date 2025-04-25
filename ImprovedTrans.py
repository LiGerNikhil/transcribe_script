import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, QWidget, 
                             QProgressBar, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap
import whisper
import torch
from datetime import datetime
import os

class TranscriptionThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model, audio_path, language, use_fp16):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        self.language = language
        self.use_fp16 = use_fp16

    def run(self):
        try:
            options = {
                "language": self.language,
                "fp16": self.use_fp16,
            }
            options = {k: v for k, v in options.items() if v is not None}
            
            result = self.model.transcribe(self.audio_path, **options)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class WhisperTranscriberGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Transcriber Pro")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))  # Add your icon file
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        # Set stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QComboBox, QCheckBox {
                padding: 5px;
            }
        """)
        
        self.init_ui()
        self.model = None
        self.transcription_thread = None

    def init_ui(self):
        # Header with logo
        header = QHBoxLayout()
        logo = QLabel()
        pixmap = QPixmap("logo.png").scaled(100, 100, Qt.KeepAspectRatio)  # Add your logo
        logo.setPixmap(pixmap)
        title = QLabel("Audio Transcriber Pro")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        header.addWidget(logo)
        header.addWidget(title)
        header.addStretch()
        self.layout.addLayout(header)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(QFont("Arial", 10))
        file_btn = QPushButton("Select Audio File")
        file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_btn)
        self.layout.addLayout(file_layout)
        
        # Settings
        settings_layout = QHBoxLayout()
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model Size:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        
        # Language selection
        lang_layout = QVBoxLayout()
        lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "en", "hi", "es", "fr", "de"])  # Add more as needed
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        settings_layout.addLayout(lang_layout)
        
        # GPU acceleration
        self.gpu_check = QCheckBox("Use GPU Acceleration")
        self.gpu_check.setChecked(True)
        self.gpu_check.setEnabled(torch.cuda.is_available())
        if not torch.cuda.is_available():
            self.gpu_check.setToolTip("CUDA not available on this system")
        settings_layout.addWidget(self.gpu_check)
        
        self.layout.addLayout(settings_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        self.layout.addWidget(self.progress)
        
        # Transcribe button
        self.transcribe_btn = QPushButton("Transcribe Audio")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(False)
        self.layout.addWidget(self.transcribe_btn)
        
        # Results display
        self.results_label = QLabel("Transcription Results:")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.layout.addWidget(self.results_label)
        self.layout.addWidget(self.results_text)
        
        # Save button
        self.save_btn = QPushButton("Save Transcription")
        self.save_btn.clicked.connect(self.save_transcription)
        self.save_btn.setEnabled(False)
        self.layout.addWidget(self.save_btn)
        
        # Status bar
        self.statusBar().showMessage("Ready")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.m4a *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.transcribe_btn.setEnabled(True)
            self.statusBar().showMessage(f"Selected: {file_path}")

    def start_transcription(self):
        if not hasattr(self, 'file_path'):
            self.statusBar().showMessage("No file selected!")
            return
            
        self.statusBar().showMessage("Loading model...")
        self.progress.setValue(0)
        self.progress.setFormat("Loading model...")
        self.transcribe_btn.setEnabled(False)
        
        try:
            device = "cuda" if self.gpu_check.isChecked() and torch.cuda.is_available() else "cpu"
            use_fp16 = device == "cuda"
            
            language = self.lang_combo.currentText()
            language = None if language == "auto" else language
            
            # Load model in background to keep UI responsive
            self.model = whisper.load_model(
                self.model_combo.currentText(), 
                device=device
            )
            
            if use_fp16:
                self.model = self.model.half()
                
            self.statusBar().showMessage("Starting transcription...")
            self.progress.setFormat("Transcribing...")
            
            # Start transcription thread
            self.transcription_thread = TranscriptionThread(
                self.model, 
                self.file_path, 
                language,
                use_fp16
            )
            self.transcription_thread.update_progress.connect(self.update_progress)
            self.transcription_thread.finished.connect(self.transcription_complete)
            self.transcription_thread.error.connect(self.transcription_error)
            self.transcription_thread.start()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            self.progress.setValue(0)
            self.transcribe_btn.setEnabled(True)

    def update_progress(self, value):
        self.progress.setValue(value)

    def transcription_complete(self, result):
        self.progress.setValue(100)
        self.progress.setFormat("Complete!")
        self.statusBar().showMessage("Transcription completed successfully")
        self.transcribe_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Format and display results
        transcript = ""
        for seg in result["segments"]:
            start = datetime.utcfromtimestamp(seg['start']).strftime('%H:%M:%S')
            end = datetime.utcfromtimestamp(seg['end']).strftime('%H:%M:%S')
            transcript += f"[{start} - {end}] {seg['text'].strip()}\n\n"
        
        self.results_text.setPlainText(transcript)
        self.transcription_result = transcript

    def transcription_error(self, error_msg):
        self.statusBar().showMessage(f"Error: {error_msg}")
        self.progress.setValue(0)
        self.progress.setFormat("Error occurred")
        self.transcribe_btn.setEnabled(True)

    def save_transcription(self):
        if not hasattr(self, 'transcription_result'):
            return
            
        default_name = f"{os.path.splitext(self.file_path)[0]}_transcript.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transcription", default_name, 
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Transcript: {os.path.basename(self.file_path)}\n")
                    f.write(f"Model: {self.model_combo.currentText()} | ")
                    f.write(f"Language: {self.lang_combo.currentText()}\n")
                    f.write("="*50 + "\n\n")
                    f.write(self.transcription_result)
                    
                self.statusBar().showMessage(f"Transcript saved to {file_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Save failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style for modern look
    app.setStyle("Fusion")
    
    window = WhisperTranscriberGUI()
    window.show()
    sys.exit(app.exec_())
