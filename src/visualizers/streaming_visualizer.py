import cv2
import numpy as np
from typing import Dict, List
from src.structures.analysis import SegmentAnalysis
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class StreamingVisualizer:
    """
    Classe para visualização em tempo real das análises de streaming.
    """
    def __init__(self, window_name: str = "Análise Multimodal"):
        """
        Inicializa o visualizador.
        
        Args:
            window_name: Nome da janela de visualização
        """
        self.window_name = window_name
        self.speaker_colors = {}  # Mapeia speaker_id para cores
        self.emotions = [
            "feliz",
            "triste",
            "raiva",
            "medo",
            "surpresa",
            "nojo",
            "neutro"
        ]
        self.emotion_colors = {
            "feliz": (0, 255, 0),      # Verde
            "triste": (255, 0, 0),     # Azul
            "raiva": (0, 0, 255),      # Vermelho
            "medo": (128, 0, 128),     # Roxo
            "surpresa": (255, 255, 0), # Amarelo
            "nojo": (0, 128, 0),       # Verde escuro
            "neutro": (128, 128, 128)  # Cinza
        }
        
        # Cores disponíveis para speakers
        self.available_colors = [
            (255, 0, 0),    # Vermelho
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarelo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Ciano
            (255, 128, 0),  # Laranja
            (128, 0, 255)   # Roxo
        ]
        
        # Inicializa o MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 para rostos a 2m, 1 para rostos a 5m
            min_detection_confidence=0.5
        )
        
    def _get_speaker_color(self, speaker_id: str) -> tuple:
        """Retorna uma cor única para cada speaker"""
        if speaker_id not in self.speaker_colors:
            # Usa a próxima cor disponível
            color_idx = len(self.speaker_colors) % len(self.available_colors)
            self.speaker_colors[speaker_id] = self.available_colors[color_idx]
        return self.speaker_colors[speaker_id]

    def draw_face_analysis(
        self,
        frame: np.ndarray,
        face_analysis: Dict,
        face_location: tuple,
        speaker_id: str
    ) -> None:
        """
        Desenha informações da análise facial ao redor do rosto detectado.
        
        Args:
            frame: Frame do vídeo
            face_analysis: Resultados da análise facial
            face_location: (x, y, w, h) do rosto
            speaker_id: Identificador do speaker
        """
        x, y, w, h = face_location
        color = self._get_speaker_color(speaker_id)
        
        # Desenha retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Desenha informações acima do rosto
        info_y = max(0, y - 10)
        cv2.putText(
            frame,
            f"Speaker: {speaker_id}",
            (x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Desenha micro-expressões
        cv2.putText(
            frame,
            f"Micro-exp: {face_analysis.get('micro_expressions', 'N/A')}",
            (x, info_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Desenha direção do olhar
        cv2.putText(
            frame,
            f"Olhar: {face_analysis.get('gaze_direction', 'N/A')}",
            (x, info_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Desenha tensão muscular
        cv2.putText(
            frame,
            f"Tensão: {face_analysis.get('muscle_tension', 'N/A')}",
            (x, info_y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    def draw_emotion_bars(
        self,
        frame: np.ndarray,
        emotions: Dict[str, float],
        x: int,
        y: int,
        speaker_id: str,
        width: int = 200,
        height: int = 20
    ) -> None:
        """
        Desenha barras de emoções no canto inferior direito.
        
        Args:
            frame: Frame do vídeo
            emotions: Dicionário de emoções e seus valores
            x: Posição x inicial
            y: Posição y inicial
            width: Largura das barras
            height: Altura das barras
            speaker_id: Identificador do speaker
        """
        color = self._get_speaker_color(speaker_id)
        
        for i, (emotion, value) in enumerate(emotions.items()):
            # Barra de fundo
            cv2.rectangle(
                frame,
                (x, y + i * (height + 5)),
                (x + width, y + i * (height + 5) + height),
                (200, 200, 200),
                -1
            )
            
            # Barra de valor
            bar_width = int(value * width)
            cv2.rectangle(
                frame,
                (x, y + i * (height + 5)),
                (x + bar_width, y + i * (height + 5) + height),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                frame,
                f"{emotion}: {value:.2f}",
                (x + 5, y + i * (height + 5) + height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    def draw_audio_analysis(
        self,
        frame: np.ndarray,
        audio_analysis: Dict,
        x: int,
        y: int,
        speaker_id: str
    ) -> None:
        """
        Desenha informações da análise de áudio no canto superior direito.
        
        Args:
            frame: Frame do vídeo
            audio_analysis: Resultados da análise de áudio
            x: Posição x inicial
            y: Posição y inicial
            speaker_id: Identificador do speaker
        """
        color = self._get_speaker_color(speaker_id)
        
        # Título
        cv2.putText(
            frame,
            "Análise de Áudio",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Pitch
        cv2.putText(
            frame,
            f"Pitch: {audio_analysis.get('pitch', 'N/A')}",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Intensidade
        cv2.putText(
            frame,
            f"Intensidade: {audio_analysis.get('intensity', 'N/A')}",
            (x, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Velocidade da fala
        cv2.putText(
            frame,
            f"Velocidade: {audio_analysis.get('speech_rate', 'N/A')}",
            (x, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    def draw_text_analysis(
        self,
        frame: np.ndarray,
        text_analysis: Dict,
        x: int,
        y: int,
        speaker_id: str
    ) -> None:
        """
        Desenha informações da análise de texto no canto superior esquerdo.
        
        Args:
            frame: Frame do vídeo
            text_analysis: Resultados da análise de texto
            x: Posição x inicial
            y: Posição y inicial
            speaker_id: Identificador do speaker
        """
        color = self._get_speaker_color(speaker_id)
        
        # Título
        cv2.putText(
            frame,
            "Análise de Texto",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Sarcasmo
        cv2.putText(
            frame,
            f"Sarcasmo: {text_analysis.get('sarcasm_score', 'N/A')}",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Humor
        cv2.putText(
            frame,
            f"Humor: {text_analysis.get('humor_score', 'N/A')}",
            (x, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        
        # Polaridade
        cv2.putText(
            frame,
            f"Polaridade: {text_analysis.get('polarity', 'N/A')}",
            (x, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    def visualize(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """Visualiza os resultados da análise"""
        try:
            # --- Face ---
            if analysis["face"] is not None and analysis["face"]["face_position"] is not None:
                # Desenha bounding box
                face_position = analysis["face"]["face_position"]
                x, y, w, h = face_position['x'], face_position['y'], face_position['w'], face_position['h']
                
                # Desenha retângulo com cor baseada na emoção
                if analysis["face"]["emotion_probs"] is not None:
                    emotion_probs = np.array(analysis["face"]["emotion_probs"]).flatten()
                    if len(emotion_probs) > 0:  # Verifica se tem emoções
                        emotion_idx = np.argmax(emotion_probs)
                        if emotion_idx < len(self.emotions):  # Verifica se o índice é válido
                            emotion = self.emotions[emotion_idx]
                            color = self.emotion_colors[emotion]
                        else:
                            color = (0, 255, 0)  # Verde padrão
                    else:
                        color = (0, 255, 0)  # Verde padrão
                else:
                    color = (0, 255, 0)  # Verde padrão
                
                # Desenha retângulo com borda mais grossa
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Desenha emoções
                if analysis["face"]["emotion_probs"] is not None:
                    emotion_probs = np.array(analysis["face"]["emotion_probs"]).flatten()
                    if len(emotion_probs) > 0:  # Verifica se tem emoções
                        emotion_idx = np.argmax(emotion_probs)
                        if emotion_idx < len(self.emotions):  # Verifica se o índice é válido
                            emotion = self.emotions[emotion_idx]
                            confidence = float(emotion_probs[emotion_idx])
                            
                            # Desenha nome da emoção
                            cv2.putText(frame, f"Face: {emotion}", (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Desenha confiança
                            cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y + h + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Desenha qualidade
                if analysis["face"]["face_quality"] is not None:
                    face_quality = analysis["face"]["face_quality"]
                    cv2.putText(frame, f"Qual: {face_quality['detection_confidence']:.2f}", 
                              (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Áudio ---
            if analysis["audio"] is not None and analysis["audio"]["emotion_probs"] is not None:
                # Desenha emoções
                emotion_probs = np.array(analysis["audio"]["emotion_probs"]).flatten()
                if len(emotion_probs) > 0:  # Verifica se tem emoções
                    emotion_idx = np.argmax(emotion_probs)
                    if emotion_idx < len(self.emotions):  # Verifica se o índice é válido
                        emotion = self.emotions[emotion_idx]
                        confidence = float(emotion_probs[emotion_idx])
                        color = self.emotion_colors[emotion]
                        
                        cv2.putText(frame, f"Áudio: {emotion}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Desenha qualidade
                        if analysis["audio"]["audio_quality"] is not None:
                            audio_quality = analysis["audio"]["audio_quality"]
                            cv2.putText(frame, f"Qual: {audio_quality['quality']:.2f}", (10, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Texto ---
            if analysis["text"] is not None and analysis["text"]["emotion_probs"] is not None:
                # Desenha emoções
                emotion_probs = np.array(analysis["text"]["emotion_probs"]).flatten()
                if len(emotion_probs) > 0:  # Verifica se tem emoções
                    emotion_idx = np.argmax(emotion_probs)
                    if emotion_idx < len(self.emotions):  # Verifica se o índice é válido
                        emotion = self.emotions[emotion_idx]
                        confidence = float(emotion_probs[emotion_idx])
                        color = self.emotion_colors[emotion]
                        
                        cv2.putText(frame, f"Texto: {emotion}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Desenha qualidade
                        if analysis["text"]["text_quality"] is not None:
                            text_quality = analysis["text"]["text_quality"]
                            cv2.putText(frame, f"Qual: {text_quality['quality']:.2f}", (10, 130),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Emoção fundida ---
            if analysis["fused_emotion"] is not None:
                emotion_probs = np.array(analysis["fused_emotion"]).flatten()
                if len(emotion_probs) > 0:  # Verifica se tem emoções
                    emotion_idx = np.argmax(emotion_probs)
                    if emotion_idx < len(self.emotions):  # Verifica se o índice é válido
                        emotion = self.emotions[emotion_idx]
                        confidence = float(emotion_probs[emotion_idx])
                        color = self.emotion_colors[emotion]
                        
                        cv2.putText(frame, f"Fusão: {emotion}", (10, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 170),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Speaker ID ---
            if analysis["speaker_id"] is not None:
                cv2.putText(frame, f"Speaker: {analysis['speaker_id']}", (10, 190),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return frame
        except Exception as e:
            logger.error(f"Erro na visualização: {e}", exc_info=True)
            return frame 