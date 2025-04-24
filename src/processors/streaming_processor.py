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
import traceback

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

    def process_segment(self, video_frames: List[np.ndarray], audio_data: bytes, text: str) -> Dict:
        """Processa um segmento de dados"""
        try:
            # --- Helper para garantir batch dimension e torch.Tensor ---
            def ensure_batch(tensor):
                if tensor is None:
                    return None
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, device=self.device)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                return tensor

            # Converte audio_data de bytes para numpy array
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            except Exception as e:
                logger.error(f"Erro ao converter áudio: {e}", exc_info=True)
                return { "face": None, "audio": None, "text": None, "fused_emotion": None, "weights": None, "speaker_id": None }

            # Salva áudio temporariamente
            audio_path = 'temp_audio.wav'
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            # --- Diarização ---
            try:
                diarization = self.diarization_pipeline(audio_path)
                speaker_id = next((speaker for turn, _, speaker in diarization.itertracks(yield_label=True)
                                   if turn.start <= 0 and turn.end >= len(audio_array)/self.sample_rate), "unknown")
            except Exception as e:
                logger.warning(f"Erro na diarização: {e}", exc_info=True)
                speaker_id = "unknown"

            # --- Unimodal analyses ---
            face_results = audio_results = text_results = None

            # Facial
            try:
                face_results = self.face_analyzer.analyze(video_frames[0], speaker_id)
            except Exception as e:
                logger.error(f"Erro na análise facial: {e}", exc_info=True)

            # Áudio
            if audio_array.size > 0:
                try:
                    audio_results = self.audio_analyzer.analyze(audio_path, speaker_id)
                except Exception as e:
                    logger.error(f"Erro na análise de áudio: {e}", exc_info=True)

            # Texto
            if text and text.strip():
                try:
                    text_results = self.text_analyzer.analyze(text, speaker_id)
                except Exception as e:
                    logger.error(f"Erro na análise de texto: {e}", exc_info=True)

            # --- Montagem de features com concateração explícita em dim=1 ---
            face_features = None
            if face_results is not None:
                face_pos_tensor = torch.tensor([
                    face_results.face_position['x'],
                    face_results.face_position['y'],
                    face_results.face_position['w'],
                    face_results.face_position['h']
                ], device=self.device).float()
                pieces = [
                    face_results.emotion_probs.float(),
                    face_results.micro_expressions.float(),
                    face_results.gaze_direction.float(),
                    face_results.muscle_tension.float(),
                    face_results.movement_patterns.float(),
                    face_pos_tensor
                ]
                pieces = [ensure_batch(p) for p in pieces]
                face_features = torch.cat(pieces, dim=1)  # shape [1,27]

            audio_features = None
            if audio_results is not None:
                audio_quality_tensor = torch.tensor([
                    audio_results.audio_quality,
                    audio_results.signal_noise_ratio,
                    audio_results.clarity,
                    audio_results.consistency
                ], device=self.device).float()
                pieces = [
                    audio_results.emotion_probs.float(),
                    audio_results.pitch.float(),
                    audio_results.intensity.float(),
                    audio_results.timbre.float(),
                    audio_results.speech_rate.float(),
                    audio_results.rhythm.float(),
                    audio_quality_tensor
                ]
                pieces = [ensure_batch(p) for p in pieces]
                audio_features = torch.cat(pieces, dim=1)  # shape [1,31]

            text_features = None
            if text_results is not None:
                text_quality_tensor = torch.tensor([
                    text_results.text_quality,
                    text_results.coherence,
                    text_results.completeness,
                    text_results.relevance
                ], device=self.device).float()
                pieces = [
                    text_results.emotion_probs.float(),
                    text_results.sarcasm_score.float(),
                    text_results.humor_score.float(),
                    text_results.polarity.float(),
                    text_results.intensity.float(),
                    text_results.context_embedding.float(),
                    text_quality_tensor
                ]
                pieces = [ensure_batch(p) for p in pieces]
                text_features = torch.cat(pieces, dim=1)  # shape [1,783]

            # --- Fusão ---
            fused_results = None
            weights = None

            # Remove nan das features para cada modalidade
            if face_features is not None:
                face_features = torch.nan_to_num(face_features, nan=0.0)
            if audio_features is not None:
                audio_features = torch.nan_to_num(audio_features, nan=0.0)
            if text_features is not None:
                text_features = torch.nan_to_num(text_features, nan=0.0)

            with torch.no_grad():
                fused_results = self.fusion_model(face_features, audio_features, text_features)
                weights = self.fusion_model.get_weights()

            # Extrai explicitamente o tensor de fusão (ou o fallback mais confiável)
            fused_tensor = None
            if fused_results is not None:
                if fused_results.get("fused") is not None:
                    fused_tensor = fused_results["fused"]
                elif fused_results.get("face") is not None:
                    fused_tensor = fused_results["face"]
                elif fused_results.get("audio") is not None:
                    fused_tensor = fused_results["audio"]
                else:
                    fused_tensor = fused_results.get("text")

                # Converte para numpy se tiver um tensor
                if fused_tensor is not None:
                    fused_tensor = fused_tensor.detach().cpu().numpy()

            # --- Prepara saída ---
            output = {
                "face": {
                    "emotion_probs": face_results.emotion_probs.detach().cpu().numpy().squeeze() if face_results else None,
                    "micro_expressions": face_results.micro_expressions.detach().cpu().numpy().squeeze() if face_results else None,
                    "gaze_direction": face_results.gaze_direction.detach().cpu().numpy().squeeze() if face_results else None,
                    "muscle_tension": face_results.muscle_tension.detach().cpu().numpy().squeeze() if face_results else None,
                    "movement_patterns": face_results.movement_patterns.detach().cpu().numpy().squeeze() if face_results else None,
                    "face_position": face_results.face_position if face_results else None,
                    "face_quality": {
                        "detection_confidence": face_results.detection_confidence,
                        "landmark_quality":   face_results.landmark_quality,
                        "expression_quality": face_results.expression_quality,
                        "movement_quality":   face_results.movement_quality
                    } if face_results else None
                },
                "audio": {
                    "emotion_probs": audio_results.emotion_probs.detach().cpu().numpy().squeeze() if audio_results else None,
                    "pitch":         audio_results.pitch.detach().cpu().numpy().squeeze() if audio_results else None,
                    "intensity":     audio_results.intensity.detach().cpu().numpy().squeeze() if audio_results else None,
                    "timbre":        audio_results.timbre.detach().cpu().numpy().squeeze() if audio_results else None,
                    "speech_rate":   audio_results.speech_rate.detach().cpu().numpy().squeeze() if audio_results else None,
                    "rhythm":        audio_results.rhythm.detach().cpu().numpy().squeeze() if audio_results else None,
                    "audio_quality": {
                        "quality":             audio_results.audio_quality,
                        "signal_noise_ratio":  audio_results.signal_noise_ratio,
                        "clarity":             audio_results.clarity,
                        "consistency":         audio_results.consistency
                    } if audio_results else None
                },
                "text": {
                    "emotion_probs":   text_results.emotion_probs.detach().cpu().numpy().squeeze() if text_results else None,
                    "sarcasm_score":   text_results.sarcasm_score.detach().cpu().numpy().squeeze() if text_results else None,
                    "humor_score":     text_results.humor_score.detach().cpu().numpy().squeeze() if text_results else None,
                    "polarity":        text_results.polarity.detach().cpu().numpy().squeeze() if text_results else None,
                    "intensity":       text_results.intensity.detach().cpu().numpy().squeeze() if text_results else None,
                    "context_embedding": text_results.context_embedding.detach().cpu().numpy().squeeze() if text_results else None,
                    "text_quality": {
                        "quality":    text_results.text_quality,
                        "coherence":  text_results.coherence,
                        "completeness": text_results.completeness,
                        "relevance":    text_results.relevance
                    } if text_results else None
                },
                "fused_emotion": fused_tensor.squeeze() if fused_tensor is not None else None,
                "weights":       weights,
                "speaker_id":    speaker_id
            }
            return output

        except Exception as e:
            logger.error(f"Erro no processamento do segmento: {e}", exc_info=True)
            return {
                "face": None, "audio": None, "text": None,
                "fused_emotion": None, "weights": None, "speaker_id": None
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
                            logger.error(f"Erro durante o processamento: {str(e)}", exc_info=True)
                            logger.error(f"Tipo do erro: {type(e).__name__}")
                            logger.error(f"Stack trace completo:", exc_info=True)
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
            logger.error(f"Erro durante o processamento: {str(e)}", exc_info=True)
            logger.error(f"Tipo do erro: {type(e).__name__}")
            logger.error(f"Stack trace completo:", exc_info=True)
        finally:
            self.stop_capture()
            cv2.destroyAllWindows() 