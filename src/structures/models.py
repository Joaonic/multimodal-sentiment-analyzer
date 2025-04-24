from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class EmotionVector:
    """Vetor de emoções"""
    neutral: float
    happy: float
    sad: float
    angry: float
    fearful: float
    disgusted: float
    surprised: float

    def to_tensor(self) -> torch.Tensor:
        """Converte para tensor"""
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
        """Cria a partir de um tensor"""
        return cls(
            neutral=tensor[0].item(),
            happy=tensor[1].item(),
            sad=tensor[2].item(),
            angry=tensor[3].item(),
            fearful=tensor[4].item(),
            disgusted=tensor[5].item(),
            surprised=tensor[6].item()
        )

@dataclass
class AudioFeatures:
    """Características de áudio"""
    emotion_vector: EmotionVector
    pitch: float
    intensity: float
    timbre: np.ndarray
    speech_rate: float
    rhythm: float

@dataclass
class TextFeatures:
    """Características de texto"""
    emotion_vector: EmotionVector
    sarcasm_score: float
    humor_score: float
    polarity: float
    intensity: float
    context_embedding: np.ndarray

@dataclass
class FaceFeatures:
    """Características faciais"""
    emotion_vector: EmotionVector
    micro_expressions: np.ndarray
    gaze_direction: np.ndarray
    muscle_tension: float

@dataclass
class FusionInput:
    """Entrada para o modelo de fusão"""
    audio: AudioFeatures
    text: TextFeatures
    face: FaceFeatures

@dataclass
class FusionOutput:
    """Saída do modelo de fusão"""
    emotion_vector: EmotionVector
    confidence: float
    modality_weights: Dict[str, float] 