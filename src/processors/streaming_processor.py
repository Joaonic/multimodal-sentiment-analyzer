import json
import logging
import os
import queue
import time
import wave
from typing import Dict, List, Optional, Callable, Tuple

import cv2
import numpy as np
import pyaudio
import torch
from pyannote.audio import Pipeline

from src.analyzers import FaceAnalyzer, TextAnalyzer, AudioAnalyzer
from src.models import AdvancedFusionModel
from src.visualizers import StreamingVisualizer
from src.structures import (
    StreamingConfig,
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

class StreamingProcessor:
    """
    Classe para processamento de streaming em tempo real.
    """
    def __init__(
        self,
        model_config: Dict,
        streaming_config: StreamingConfig
    ):
        """
        Inicializa o processador de streaming.
        
        Args:
            model_config: Configurações dos modelos
            streaming_config: Configurações de streaming
        """
        self.model_config = model_config
        self.streaming_config = streaming_config
        
        # Define o dispositivo a ser usado (CPU ou CUDA)
        self.device = torch.device(model_config['device'])
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializa modelos
        self.face_analyzer = FaceAnalyzer(device=self.device)
        self.audio_analyzer = AudioAnalyzer(device=self.device)
        self.text_analyzer = TextAnalyzer(device=self.device)
        
        self.fusion_model = AdvancedFusionModel.load(
            model_config['fusion_model'],
            self.device
        )
        
        # Inicializa pipeline de diarização
        logger.info(f"Inicializando pipeline de diarização com token: {model_config['hf_token'][:5]}...{model_config['hf_token'][-5:]}")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=model_config['hf_token']
        )
        logger.info("Pipeline de diarização inicializado com sucesso")
        
        # Inicializa visualizador
        self.visualizer = StreamingVisualizer()
        
        # Configurações de áudio
        self.audio_format = pyaudio.paInt16
        self.channels = streaming_config['channels']
        self.sample_rate = streaming_config['sample_rate']
        self.chunk_size = streaming_config['chunk_size']
        
        # Buffers
        self.audio_buffer = queue.Queue()
        self.video_buffer = []
        self.max_buffer_size = 30  # Limita o tamanho do buffer de vídeo
        
        # Flags
        self.is_running = False
        self.current_speaker = "stream_speaker"
        
        # Cores para emoções
        self.emotion_colors = {
            "happy": (0, 255, 0),      # Verde
            "sad": (255, 0, 0),        # Azul
            "angry": (0, 0, 255),      # Vermelho
            "fear": (128, 0, 128),     # Roxo
            "surprise": (255, 255, 0), # Amarelo
            "disgust": (0, 128, 0),    # Verde escuro
            "neutral": (128, 128, 128) # Cinza
        }

    def audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Dict,
        status: int
    ) -> Tuple[bytes, int]:
        """
        Callback para captura de áudio.
        """
        self.audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start_capture(self):
        """
        Inicia a captura de áudio e vídeo.
        """
        # Inicializa captura de áudio
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        # Inicializa captura de vídeo
        self.video_capture = cv2.VideoCapture(self.streaming_config['video_source'])
        
        # Configurações do vídeo
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        
        # Inicializa writer de vídeo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            'temp_stream.mp4',
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        self.is_running = True
        logger.info("Captura iniciada")

    def stop_capture(self):
        """
        Para a captura de áudio e vídeo.
        """
        self.is_running = False
        
        # Para captura de áudio
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
        
        # Para captura de vídeo
        self.video_capture.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
        logger.info("Captura encerrada")

    def process_segment(self, video_frames: List[np.ndarray], audio_data: np.ndarray, text: str) -> Dict:
        """Processa um segmento de dados"""
        try:
            # Salva áudio temporário
            audio_path = 'temp_audio.wav'
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            # Identifica falante usando diarização
            try:
                diarization = self.diarization_pipeline(audio_path)
                speaker_id = None
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= 0 and turn.end >= len(audio_data)/self.sample_rate:
                        speaker_id = speaker
                        break
            except Exception as e:
                logger.warning(f"Erro na diarização: {e}")
                speaker_id = "unknown"

            # Processa cada modalidade
            face_results = self.face_analyzer.analyze(video_frames[0], speaker_id)  # Usa o primeiro frame
            audio_results = self.audio_analyzer.analyze(audio_path, speaker_id)
            text_results = self.text_analyzer.analyze(text, speaker_id)
            
            # Prepara os tensores para o modelo de fusão
            # Face: 7 emoções + 5 micro_expressions + 3 gaze_direction + 4 muscle_tension + 4 movement_patterns + 4 face_position
            face_features = torch.cat([
                face_results.emotion_probs.float(),
                face_results.micro_expressions.float(),
                face_results.gaze_direction.float(),
                face_results.muscle_tension.float(),
                face_results.movement_patterns.float(),
                torch.tensor([
                    face_results.face_position['x'],
                    face_results.face_position['y'],
                    face_results.face_position['w'],
                    face_results.face_position['h']
                ], device=self.device).float()
            ]).unsqueeze(0)  # [1, 27]
            
            # Audio: 8 emoções + pitch + intensity + timbre + speech_rate + rhythm + 4 audio_quality
            audio_features = torch.cat([
                audio_results.emotion_probs.float(),
                audio_results.pitch.float(),
                audio_results.intensity.float(),
                audio_results.timbre.float(),
                audio_results.speech_rate.float(),
                audio_results.rhythm.float(),
                torch.tensor([
                    audio_results.audio_quality,
                    audio_results.signal_noise_ratio,
                    audio_results.clarity,
                    audio_results.consistency
                ], device=self.device).float()
            ]).unsqueeze(0)  # [1, 30]
            
            # Text: 7 emoções + sarcasm + humor + polarity + intensity + context_embedding + 4 text_quality
            text_features = torch.cat([
                text_results.emotion_probs.float(),
                text_results.sarcasm_score.float(),
                text_results.humor_score.float(),
                text_results.polarity.float(),
                text_results.intensity.float(),
                text_results.context_embedding.float(),
                torch.tensor([
                    text_results.text_quality,
                    text_results.coherence,
                    text_results.completeness,
                    text_results.relevance
                ], device=self.device).float()
            ]).unsqueeze(0)  # [1, 783]
            
            # Garante que os tensores têm as dimensões corretas [batch, features]
            if face_features.dim() == 1:
                face_features = face_features.unsqueeze(0)
            if audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)
            
            # Faz a fusão
            with torch.no_grad():
                fused_results = self.fusion_model(face_features, audio_features, text_features)
            
            # Converte os resultados para numpy
            fused_emotion = np.array(list(fused_results.values()))
            weights = np.array(list(self.fusion_model.get_weights().values()))
            
            # Limpa arquivo temporário
            os.remove(audio_path)
            
            # Cria o dicionário de resultados
            result = {
                "face": {
                    "emotion_probs": face_results.emotion_probs.cpu().numpy(),
                    "micro_expressions": face_results.micro_expressions.cpu().numpy(),
                    "gaze_direction": face_results.gaze_direction.cpu().numpy(),
                    "muscle_tension": face_results.muscle_tension.cpu().numpy(),
                    "movement_patterns": face_results.movement_patterns.cpu().numpy(),
                    "face_position": face_results.face_position,
                    "face_quality": {
                        "detection_confidence": face_results.detection_confidence,
                        "landmark_quality": face_results.landmark_quality,
                        "expression_quality": face_results.expression_quality,
                        "movement_quality": face_results.movement_quality
                    }
                },
                "audio": {
                    "emotion_probs": audio_results.emotion_probs.cpu().numpy(),
                    "pitch": audio_results.pitch.cpu().numpy(),
                    "intensity": audio_results.intensity.cpu().numpy(),
                    "timbre": audio_results.timbre.cpu().numpy(),
                    "speech_rate": audio_results.speech_rate.cpu().numpy(),
                    "rhythm": audio_results.rhythm.cpu().numpy(),
                    "audio_quality": {
                        "quality": audio_results.audio_quality,
                        "signal_noise_ratio": audio_results.signal_noise_ratio,
                        "clarity": audio_results.clarity,
                        "consistency": audio_results.consistency
                    }
                },
                "text": {
                    "emotion_probs": text_results.emotion_probs.cpu().numpy(),
                    "sarcasm_score": text_results.sarcasm_score.cpu().numpy(),
                    "humor_score": text_results.humor_score.cpu().numpy(),
                    "polarity": text_results.polarity.cpu().numpy(),
                    "intensity": text_results.intensity.cpu().numpy(),
                    "context_embedding": text_results.context_embedding.cpu().numpy(),
                    "text_quality": {
                        "quality": text_results.text_quality,
                        "coherence": text_results.coherence,
                        "completeness": text_results.completeness,
                        "relevance": text_results.relevance
                    }
                },
                "fused_emotion": fused_emotion,
                "weights": weights,
                "speaker_id": speaker_id
            }
            
            return result
        except Exception as e:
            print(f"Erro no processamento do segmento: {e}")
            return {
                "face": {
                    "emotion_probs": np.zeros(7),
                    "micro_expressions": np.zeros(5),
                    "gaze_direction": np.zeros(3),
                    "muscle_tension": np.zeros(4),
                    "movement_patterns": np.zeros(4),
                    "face_position": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "face_quality": {
                        "detection_confidence": 0.0,
                        "landmark_quality": 0.0,
                        "expression_quality": 0.0,
                        "movement_quality": 0.0
                    }
                },
                "audio": {
                    "emotion_probs": np.zeros(8),
                    "pitch": np.zeros(1),
                    "intensity": np.zeros(1),
                    "timbre": np.zeros(13),
                    "speech_rate": np.zeros(1),
                    "rhythm": np.zeros(3),
                    "audio_quality": {
                        "quality": 0.0,
                        "signal_noise_ratio": 0.0,
                        "clarity": 0.0,
                        "consistency": 0.0
                    }
                },
                "text": {
                    "emotion_probs": np.zeros(7),
                    "sarcasm_score": np.zeros(1),
                    "humor_score": np.zeros(1),
                    "polarity": np.zeros(1),
                    "intensity": np.zeros(1),
                    "context_embedding": np.zeros(768),
                    "text_quality": {
                        "quality": 0.0,
                        "coherence": 0.0,
                        "completeness": 0.0,
                        "relevance": 0.0
                    }
                },
                "fused_emotion": np.zeros(7),
                "weights": np.zeros(3),
                "speaker_id": "unknown"
            }

    def run(
        self,
        duration: float = 5.0,
        callback: Optional[Callable] = None
    ):
        """
        Executa o processamento de streaming.
        
        Args:
            duration: Duração de cada segmento em segundos
            callback: Função para processar resultados
        """
        self.start_capture()
        start_time = time.time()
        
        # Cria uma única janela
        cv2.namedWindow(self.visualizer.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.is_running:
                # Captura frames
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning("Não foi possível capturar frame")
                    continue
                    
                # Limita o tamanho do buffer de vídeo
                if len(self.video_buffer) >= self.max_buffer_size:
                    self.video_buffer.pop(0)
                self.video_buffer.append(frame)
                
                # Processa segmento quando atingir a duração
                if time.time() - start_time >= duration:
                    # Coleta áudio do buffer
                    audio_frames = []
                    while not self.audio_buffer.empty():
                        audio_frames.append(self.audio_buffer.get())
                        
                    if audio_frames:
                        try:
                            # Processa segmento
                            result = self.process_segment(
                                self.video_buffer,
                                b''.join(audio_frames),
                                ""
                            )
                            
                            # Visualiza resultados
                            vis_frame = self.visualizer.visualize(frame, result)
                            cv2.imshow(self.visualizer.window_name, vis_frame)
                            
                            # Chama callback
                            if callback:
                                callback(result)
                        except Exception as e:
                            logger.error(f"Erro durante o processamento: {e}")
                            cv2.imshow(self.visualizer.window_name, frame)
                    else:
                        cv2.imshow(self.visualizer.window_name, frame)
                        
                    # Limpa buffers
                    self.video_buffer = []
                    start_time = time.time()
                else:
                    cv2.imshow(self.visualizer.window_name, frame)
                
                # Verifica se usuário quer sair
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Processamento interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro durante o processamento: {e}")
        finally:
            self.stop_capture()
            cv2.destroyAllWindows() 