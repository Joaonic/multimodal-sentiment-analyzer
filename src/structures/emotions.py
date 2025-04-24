from dataclasses import dataclass
from typing import List

import torch


@dataclass
class EmotionVector:
    """Vetor de emoções básicas"""
    neutral: float
    happy: float
    sad: float
    angry: float
    fearful: float
    disgusted: float
    surprised: float

    def to_tensor(self) -> torch.Tensor:
        """Converte para tensor PyTorch"""
        return torch.tensor([
            self.neutral,
            self.happy,
            self.sad,
            self.angry,
            self.fearful,
            self.disgusted,
            self.surprised
        ])

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'EmotionVector':
        """Cria a partir de um tensor PyTorch"""
        return cls(*tensor.tolist())

@dataclass
class AudioEmotionVector(EmotionVector):
    """Vetor de emoções de áudio com características adicionais"""
    pitch: float
    intensity: float
    timbre: List[float]
    speech_rate: float
    rhythm: List[float]

@dataclass
class FaceEmotionVector(EmotionVector):
    """Vetor de emoções faciais com características adicionais"""
    micro_expressions: List[float]
    gaze_direction: List[float]
    muscle_tension: List[float]
    movement_patterns: List[float]

@dataclass
class TextEmotionVector(EmotionVector):
    """Vetor de emoções textuais com características adicionais"""
    sarcasm_score: float
    humor_score: float
    polarity: float
    intensity: float
    context_embedding: List[float]

@dataclass
class FusedEmotionVector(EmotionVector):
    """Vetor de emoções fundidas com confiança"""
    confidence: float
    face_weight: float
    audio_weight: float
    text_weight: float 