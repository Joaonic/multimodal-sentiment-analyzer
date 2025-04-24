import json
import logging
import os
import subprocess
from typing import Dict, List

import cv2
import numpy as np
import torch
from pyannote.audio import Pipeline

from src.analyzers import FaceAnalyzer, TextAnalyzer, AudioAnalyzer
from src.models import AdvancedFusionModel
from src.structures import (
    ProcessingConfig,
    AnalysisResult,
    CompleteAnalysisResult,
    SegmentAnalysis
)
from src.structures.analysis import AudioAnalysis

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineProcessor:
    """
    Classe para processamento offline de vídeos.
    """
    def __init__(
        self,
        model_config: Dict,
        processing_config: ProcessingConfig,
        hf_token: str
    ):
        """
        Inicializa o processador offline.
        
        Args:
            model_config: Configurações dos modelos
            processing_config: Configurações de processamento
            hf_token: Token do HuggingFace para diarização
        """
        self.model_config = model_config
        self.processing_config = processing_config
        self.hf_token = hf_token
        
        # Inicializa modelos
        self.face_analyzer = FaceAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.text_analyzer = TextAnalyzer()
        
        self.fusion_model = AdvancedFusionModel.load(
            model_config['fusion_model'],
            model_config['device']
        )
        
        # Inicializa pipeline de diarização
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
        self.diarization_pipeline.to(torch.device(model_config['device']))
        
        # Cria diretórios necessários
        os.makedirs(processing_config['output_dir'], exist_ok=True)
        os.makedirs(processing_config['temp_dir'], exist_ok=True)

    def extract_audio(self, video_path: str) -> str:
        """
        Extrai áudio do vídeo.
        
        Args:
            video_path: Caminho do vídeo
            
        Returns:
            Caminho do arquivo de áudio extraído
        """
        audio_path = os.path.join(
            self.processing_config['temp_dir'],
            'extracted_audio.wav'
        )
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path

    def perform_diarization(self, audio_path: str) -> List[Dict]:
        """
        Realiza diarização do áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            Lista de segmentos com locutor
        """
        diarization = self.diarization_pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            
        return segments

    def extract_audio_segment(
        self,
        audio_path: str,
        start: float,
        end: float
    ) -> str:
        """
        Extrai um segmento de áudio.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            start: Início do segmento
            end: Fim do segmento
            
        Returns:
            Caminho do segmento extraído
        """
        segment_path = os.path.join(
            self.processing_config['temp_dir'],
            f'segment_{start:.2f}_{end:.2f}.wav'
        )
        
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            segment_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return segment_path

    def get_frame_at_time(self, video_path: str, time_sec: float) -> np.ndarray:
        """
        Obtém frame do vídeo em um tempo específico.
        
        Args:
            video_path: Caminho do vídeo
            time_sec: Tempo em segundos
            
        Returns:
            Frame do vídeo
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        return frame

    def process_segment(
        self,
        segment: Dict,
        video_path: str,
        audio_path: str
    ) -> SegmentAnalysis:
        """
        Processa um segmento do vídeo.
        
        Args:
            segment: Informações do segmento
            video_path: Caminho do vídeo
            audio_path: Caminho do áudio
            
        Returns:
            Análise do segmento
        """
        start, end, speaker = segment["start"], segment["end"], segment["speaker"]
        mid_time = (start + end) / 2
        
        # Obtém frame
        frame = self.get_frame_at_time(video_path, mid_time)
        
        # Extrai segmento de áudio
        audio_segment_path = self.extract_audio_segment(audio_path, start, end)
        
        # Análise facial
        face_vec = self.face_analyzer.analyze(frame)
        
        # Análise de áudio
        audio_vec = self.audio_analyzer.analyze(audio_segment_path)
        
        # Análise de texto
        text_vec = self.text_analyzer.analyze(audio_segment_path)
        
        # Fusão
        fused_vec = self.fusion_model(
            torch.tensor(list(face_vec.values())),
            torch.tensor(list(audio_vec.values())),
            torch.tensor(list(text_vec.values()))
        )
        
        # Limpa arquivo temporário
        os.remove(audio_segment_path)
        
        return {
            "start": start,
            "end": end,
            "speaker": speaker,
            "face_vec": list(face_vec.values()),
            "audio_vec": list(audio_vec.values()),
            "text_vec": list(text_vec.values()),
            "transcript": "",  # TODO
            "fused_vec": fused_vec.tolist(),
            "fused_emotion": max(fused_vec, key=fused_vec.get)
        }

    def process_video(self, video_path: str) -> List[CompleteAnalysisResult]:
        """
        Processa um vídeo completo.
        
        Args:
            video_path: Caminho do vídeo
            
        Returns:
            Lista de resultados por locutor
        """
        logger.info(f"Processando vídeo: {video_path}")
        
        # Extrai áudio
        audio_path = self.extract_audio(video_path)
        
        # Diarização
        segments = self.perform_diarization(audio_path)
        
        # Processa segmentos
        results = []
        for segment in segments:
            result = self.process_segment(segment, video_path, audio_path)
            results.append(result)
            
        # Agrupa por locutor
        speaker_results = {}
        for result in results:
            speaker = result["speaker"]
            if speaker not in speaker_results:
                speaker_results[speaker] = {
                    "person": speaker,
                    "segments": [],
                    "dominant_emotion": None,
                    "emotion_segments": [],
                    "patterns": [],
                    "raw_analysis": []
                }
                
            speaker_results[speaker]["segments"].append({
                "start": result["start"],
                "end": result["end"]
            })
            
            speaker_results[speaker]["emotion_segments"].append({
                "time": [result["start"], result["end"]],
                "emotion": result["fused_emotion"],
                "vector": result["fused_vec"]
            })
            
            speaker_results[speaker]["raw_analysis"].append(result)
            
        # Calcula emoção dominante
        for speaker in speaker_results:
            emotions = [seg["emotion"] for seg in speaker_results[speaker]["emotion_segments"]]
            dominant_emotion = max(set(emotions), key=emotions.count)
            speaker_results[speaker]["dominant_emotion"] = dominant_emotion
            
        # Detecta padrões
        for speaker in speaker_results:
            emotions = [seg["emotion"] for seg in speaker_results[speaker]["emotion_segments"]]
            for i in range(len(emotions) - 2):
                if emotions[i] == emotions[i + 1] == emotions[i + 2]:
                    pattern = f"Emoção consistente '{emotions[i]}' nos segmentos {i+1}-{i+3}"
                    speaker_results[speaker]["patterns"].append(pattern)
                    
        # Limpa arquivo temporário
        os.remove(audio_path)
        
        return list(speaker_results.values()) 