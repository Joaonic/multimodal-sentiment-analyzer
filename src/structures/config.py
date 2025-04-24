from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class ModelConfig:
    """Configurações dos modelos"""
    device: str
    face_model_name: str
    audio_model_name: str
    text_model_name: str
    fusion_model_path: Path
    face_weight: float
    audio_weight: float
    text_weight: float
    batch_size: int
    num_workers: int

@dataclass
class ProcessingConfig:
    """Configurações de processamento"""
    segment_duration: float
    min_speech_duration: float
    min_pause_duration: float
    output_dir: Path
    temp_dir: Path
    max_segments: int
    confidence_threshold: float

@dataclass
class StreamingConfig:
    """Configurações de streaming"""
    video_source: Union[int, str]
    audio_source: int
    sample_rate: int
    channels: int
    chunk_size: int
    buffer_size: int
    fps: int

@dataclass
class DirectoryConfig:
    """Configurações de diretórios"""
    data_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    output_dir: Path
    temp_dir: Path
    logs_dir: Path

@dataclass
class DiarizationConfig:
    """Configurações de diarização"""
    model: str
    min_speakers: int
    max_speakers: int
    use_auth_token: Optional[str] = None

@dataclass
class TranscriptionConfig:
    """Configurações de transcrição"""
    model: str
    language: str
    task: str
    device: Optional[str] = None

@dataclass
class FaceAnalysisConfig:
    """Configurações de análise facial"""
    backend: str
    actions: List[str]
    enforce_detection: bool
    align: bool
    detector_backend: Optional[str] = None

@dataclass
class AudioAnalysisConfig:
    """Configurações de análise de áudio"""
    sample_rate: int
    channels: int
    format: str
    window_size: float = 0.025
    hop_length: float = 0.010

@dataclass
class TextAnalysisConfig:
    """Configurações de análise de texto"""
    max_length: int
    truncation: bool
    padding: bool
    device: Optional[str] = None

@dataclass
class LoggingConfig:
    """Configurações de logging"""
    level: str
    format: str
    filename: Path
    filemode: str = 'a'

@dataclass
class SystemConfig:
    """Configurações gerais do sistema"""
    models: ModelConfig
    processing: ProcessingConfig
    streaming: StreamingConfig
    directories: DirectoryConfig
    diarization: DiarizationConfig
    transcription: TranscriptionConfig
    face_analysis: FaceAnalysisConfig
    audio_analysis: AudioAnalysisConfig
    text_analysis: TextAnalysisConfig
    logging: LoggingConfig
    debug: bool
    log_level: str 