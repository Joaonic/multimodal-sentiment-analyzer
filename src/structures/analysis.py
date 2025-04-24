from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch

from .emotions import (
    FaceEmotionVector,
    AudioEmotionVector,
    TextEmotionVector,
    FusedEmotionVector
)

class DictMixin:
    """Mixin para permitir acesso como dicionário"""
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return asdict(self)

@dataclass
class FaceAnalysis(DictMixin):
    """Resultado da análise facial"""
    speaker_id: str  # Identificador do locutor
    emotion_probs: torch.Tensor  # 7 emoções
    micro_expressions: torch.Tensor  # 5 micro-expressões
    gaze_direction: torch.Tensor    # 3 direções do olhar
    muscle_tension: torch.Tensor    # 4 tensões musculares
    movement_patterns: torch.Tensor # 4 padrões de movimento
    face_position: Dict[str, int]   # Posição da face no frame (x, y, w, h)
    # Novas características de qualidade
    detection_confidence: float     # Confiança na detecção da face
    landmark_quality: float         # Qualidade dos landmarks
    expression_quality: float       # Qualidade da detecção de expressões
    movement_quality: float         # Qualidade da detecção de movimentos

@dataclass
class AudioAnalysis(DictMixin):
    """Resultado da análise de áudio"""
    speaker_id: str  # Identificador do locutor
    emotion_probs: torch.Tensor  # 8 emoções
    pitch: torch.Tensor         # Entonação
    intensity: torch.Tensor     # Intensidade
    timbre: torch.Tensor       # Timbre
    speech_rate: torch.Tensor  # Velocidade da fala
    rhythm: torch.Tensor      # Ritmo e pausas
    # Novas características de qualidade
    audio_quality: float        # Qualidade geral do áudio
    signal_noise_ratio: float   # Relação sinal-ruído
    clarity: float             # Clareza do áudio
    consistency: float         # Consistência do áudio

@dataclass
class TextAnalysis(DictMixin):
    """Resultado da análise de texto"""
    speaker_id: str  # Identificador do locutor
    emotion_probs: torch.Tensor  # 7 emoções
    sarcasm_score: torch.Tensor  # Probabilidade de sarcasmo
    humor_score: torch.Tensor   # Probabilidade de humor
    polarity: torch.Tensor      # Polaridade (-1 a 1)
    intensity: torch.Tensor     # Intensidade emocional
    context_embedding: torch.Tensor  # Embedding contextual
    # Novas características de qualidade
    text_quality: float         # Qualidade geral do texto
    coherence: float           # Coerência do texto
    completeness: float        # Completude do texto
    relevance: float          # Relevância do texto

@dataclass
class AnalysisResult(DictMixin):
    """Resultado de uma análise de segmento"""
    start_time: float
    end_time: float
    speaker_id: str  # Identificador do locutor
    face_analysis: FaceEmotionVector
    audio_analysis: AudioEmotionVector
    text_analysis: TextEmotionVector
    fused_analysis: FusedEmotionVector
    transcript: Optional[str]
    confidence: float
    dominant_emotion: str

@dataclass
class SegmentAnalysis(DictMixin):
    """Análise de um segmento de vídeo/áudio"""
    start_time: float
    end_time: float
    speaker_id: str  # Identificador do locutor
    face_analysis: FaceEmotionVector
    audio_analysis: AudioEmotionVector
    text_analysis: TextEmotionVector
    fused_analysis: FusedEmotionVector
    transcript: Optional[str]
    confidence: float
    dominant_emotion: str

@dataclass
class SpeakerAnalysis(DictMixin):
    """Análise completa de um locutor"""
    speaker_id: str
    segments: List[SegmentAnalysis]
    dominant_emotion: str
    emotion_patterns: List[str]
    average_confidence: float
    emotion_timeline: List[Dict[str, Union[float, str]]]

@dataclass
class VideoAnalysis(DictMixin):
    """Análise completa de um vídeo"""
    video_path: Path
    duration: float
    speakers: List[SpeakerAnalysis]
    global_emotion: str
    emotion_transitions: List[Dict[str, Union[float, str]]]
    confidence: float

@dataclass
class StreamingAnalysis(DictMixin):
    """Análise em tempo real"""
    current_emotion: str
    current_confidence: float
    emotion_history: List[Dict[str, Union[float, str]]]
    speaker_id: str  # Identificador do locutor
    timestamp: float
    is_speaking: bool
    face_detected: bool
    audio_quality: float

@dataclass
class CompleteAnalysisResult(DictMixin):
    """Resultado completo da análise"""
    video_path: Path
    duration: float
    speakers: List[SpeakerAnalysis]
    global_emotion: str
    emotion_transitions: List[Dict[str, Union[float, str]]]
    confidence: float
    processing_time: float
    error: Optional[str] = None 