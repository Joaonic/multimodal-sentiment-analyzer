from typing import List, Dict, Optional, TypedDict, Union
import numpy as np

class EmotionVector(TypedDict):
    angry: float
    disgust: float
    fear: float
    happy: float
    sad: float
    surprise: float
    neutral: float

class SegmentAnalysis(TypedDict):
    start: float
    end: float
    speaker: str
    face_vec: List[float]
    audio_vec: List[float]
    text_vec: List[float]
    transcript: str
    fused_vec: List[float]
    fused_emotion: str

class SpeakerPattern(TypedDict):
    start: float
    end: float
    emotion: str

class SpeakerData(TypedDict):
    segments: List[Dict[str, float]]
    emotion_segments: List[Dict[str, Union[List[float], str, List[float]]]]
    patterns: List[str]

class AnalysisResult(TypedDict):
    person: str
    segments: List[Dict[str, float]]
    dominant_emotion: str
    emotion_segments: List[Dict[str, Union[List[float], str, List[float]]]]
    patterns: List[str]

class CompleteAnalysisResult(TypedDict):
    person: str
    segments: List[SegmentAnalysis]
    dominant_emotion: str
    emotion_segments: List[Dict[str, Union[List[float], str, List[float]]]]
    patterns: List[str]
    raw_analysis: List[SegmentAnalysis] 