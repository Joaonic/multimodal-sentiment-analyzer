import logging
from typing import List
import os
import math

import cv2
import mediapipe as mp
import numpy as np
import torch
from deepface import DeepFace
from src.structures.analysis import FaceAnalysis
from src.utils.normalization import FaceFeatureNormalizer

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.normalizer = FaceFeatureNormalizer(device)
        logger.info(f"Inicializando FaceAnalyzer no dispositivo: {device}")
        
        # Inicializa o MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configurações
        self.history_size = 10  # Número de frames para análise de movimento
        self.face_history = []
        
        # Modelo de análise facial
        self.face_model = os.getenv("FACE_MODEL", "deepface")
        logger.info(f"Modelo facial configurado: {self.face_model}")
        
    def analyze(self, frame: np.ndarray, speaker_id: str) -> FaceAnalysis:
        """Analisa o frame e retorna todas as características faciais"""
        try:
            logger.debug(f"Iniciando análise facial para speaker_id: {speaker_id}")
            
            # Detecta face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Inicializa posição padrão
            face_position = {"x": 0, "y": 0, "w": 0, "h": 0}
            
            # Inicializa métricas de qualidade
            detection_confidence = 0.0
            landmark_quality = 0.0
            expression_quality = 0.0
            movement_quality = 0.0
            
            if results.multi_face_landmarks:
                logger.debug("Face detectada, processando landmarks")
                
                # Calcula bounding box da face
                landmarks = results.multi_face_landmarks[0].landmark
                x_coords = [landmark.x for landmark in landmarks]
                y_coords = [landmark.y for landmark in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                h, w = frame.shape[:2]
                face_position = {
                    "x": int(x_min * w),
                    "y": int(y_min * h),
                    "w": int((x_max - x_min) * w),
                    "h": int((y_max - y_min) * h)
                }
                
                # Garante que a posição não saia dos limites
                face_position["x"] = max(0, min(face_position["x"], w))
                face_position["y"] = max(0, min(face_position["y"], h))
                face_position["w"] = max(0, min(face_position["w"], w - face_position["x"]))
                face_position["h"] = max(0, min(face_position["h"], h - face_position["y"]))
                
                # Calcula métricas de qualidade
                detection_confidence = self._calculate_detection_confidence(landmarks)
                landmark_quality = self._calculate_landmark_quality(landmarks)
                expression_quality = self._calculate_expression_quality(landmarks)
                movement_quality = self._calculate_movement_quality(landmarks)
                
                logger.debug(f"Métricas de qualidade calculadas - Detecção: {detection_confidence:.2f}, Landmarks: {landmark_quality:.2f}, Expressão: {expression_quality:.2f}, Movimento: {movement_quality:.2f}")
            
            # Análise de emoção básica
            logger.debug("Iniciando análise de emoção")
            emotion_probs = self._analyze_emotion(frame)
            logger.debug(f"Dimensões do tensor de emoção: {emotion_probs.shape}")
            
            # Análise de micro-expressões
            logger.debug("Iniciando análise de micro-expressões")
            micro_expressions = self._analyze_micro_expressions(frame)
            logger.debug(f"Dimensões do tensor de micro-expressões: {micro_expressions.shape}")
            
            # Análise de direção do olhar
            logger.debug("Iniciando análise de direção do olhar")
            gaze_direction = self._analyze_gaze(frame)
            logger.debug(f"Dimensões do tensor de direção do olhar: {gaze_direction.shape}")
            
            # Análise de tensão muscular
            logger.debug("Iniciando análise de tensão muscular")
            muscle_tension = self._analyze_muscle_tension(frame)
            logger.debug(f"Dimensões do tensor de tensão muscular: {muscle_tension.shape}")
            
            # Análise de padrões de movimento
            logger.debug("Iniciando análise de padrões de movimento")
            movement_patterns = self._analyze_movement(frame)
            logger.debug(f"Dimensões do tensor de padrões de movimento: {movement_patterns.shape}")
            
            # Concatena todas as features
            features = torch.cat([
                emotion_probs,
                micro_expressions,
                gaze_direction,
                muscle_tension,
                movement_patterns
            ], dim=1)
            
            # Normaliza as features
            features = self.normalizer.normalize(features)
            logger.debug(f"Dimensões após normalização: {features.shape}")
            
            return FaceAnalysis(
                speaker_id=speaker_id,
                emotion_probs=features[:, :7],         # Primeiras 7 dimensões são emoções
                micro_expressions=features[:, 7:12],    # Próximas 5 dimensões são micro-expressões
                gaze_direction=features[:, 12:15],      # Próximas 3 dimensões são direção do olhar
                muscle_tension=features[:, 15:19],      # Próximas 4 dimensões são tensão muscular
                movement_patterns=features[:, 19:23],   # Próximas 4 dimensões são padrões de movimento
                face_position=face_position,
                detection_confidence=detection_confidence,
                landmark_quality=landmark_quality,
                expression_quality=expression_quality,
                movement_quality=movement_quality
            )
        except Exception as e:
            logger.error(f"Erro na análise facial: {str(e)}", exc_info=True)
            return self._get_default_analysis(speaker_id)
    
    def _analyze_emotion(self, frame: np.ndarray) -> torch.Tensor:
        """Analisa as emoções básicas"""
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                align=True
            )
            
            if isinstance(result, list):
                result = result[0]
                
            emotion_dict = result.get("emotion", {})
            emotion_probs = torch.tensor([
                emotion_dict.get("angry", 0),
                emotion_dict.get("disgust", 0),
                emotion_dict.get("fear", 0),
                emotion_dict.get("happy", 0),
                emotion_dict.get("sad", 0),
                emotion_dict.get("surprise", 0),
                emotion_dict.get("neutral", 0)
            ], device=self.device).float()
            
            if emotion_probs.sum() > 0:
                emotion_probs = emotion_probs / emotion_probs.sum()
            else:
                emotion_probs = torch.ones(7, device=self.device).float() / 7
            
            # Garante que o tensor tem dimensão de batch
            if emotion_probs.dim() == 1:
                emotion_probs = emotion_probs.unsqueeze(0)
                
            return emotion_probs
        except Exception as e:
            print(f"Erro na análise de emoção: {e}")
            return torch.ones(1, 7, device=self.device).float() / 7
    
    def _analyze_micro_expressions(self, frame: np.ndarray) -> torch.Tensor:
        """Analisa micro-expressões usando landmarks faciais"""
        try:
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return torch.zeros(1, 5, device=self.device)  # [batch, features]
                
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calcula distâncias entre landmarks específicos
            micro_features = torch.zeros(5, device=self.device)
            
            # 1. Tensão na testa
            forehead_tension = self._calculate_distance(
                landmarks[10],  # Testa superior
                landmarks[151]  # Testa inferior
            )
            micro_features[0] = forehead_tension
            
            # 2. Movimento das sobrancelhas
            brow_movement = self._calculate_distance(
                landmarks[105],  # Sobrancelha esquerda
                landmarks[334]   # Sobrancelha direita
            )
            micro_features[1] = brow_movement
            
            # 3. Tensão ao redor dos olhos
            eye_tension = self._calculate_distance(
                landmarks[33],   # Canto externo do olho
                landmarks[133]   # Canto interno do olho
            )
            micro_features[2] = eye_tension
            
            # 4. Movimento do nariz
            nose_movement = self._calculate_distance(
                landmarks[1],    # Ponte do nariz
                landmarks[4]     # Ponta do nariz
            )
            micro_features[3] = nose_movement
            
            # 5. Tensão ao redor da boca
            mouth_tension = self._calculate_distance(
                landmarks[61],   # Canto da boca
                landmarks[291]   # Outro canto da boca
            )
            micro_features[4] = mouth_tension
            
            # Normaliza
            micro_features = (micro_features - micro_features.mean()) / (micro_features.std() + 1e-6)
            
            # Adiciona dimensão de batch
            return micro_features.unsqueeze(0)  # [1, 5]
        except Exception as e:
            print(f"Erro na análise de micro-expressões: {e}")
            return torch.zeros(1, 5, device=self.device)
    
    def _analyze_gaze(self, frame: np.ndarray) -> torch.Tensor:
        """Analisa a direção do olhar"""
        try:
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return torch.zeros(1, 3, device=self.device)  # [batch, features]
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calcula a direção do olhar usando landmarks dos olhos
            left_eye = torch.tensor([
                landmarks[33].x - landmarks[133].x,  # Horizontal
                landmarks[159].y - landmarks[145].y,  # Vertical
                landmarks[33].z - landmarks[133].z   # Profundidade
            ], device=self.device)
            
            right_eye = torch.tensor([
                landmarks[362].x - landmarks[263].x,  # Horizontal
                landmarks[386].y - landmarks[374].y,  # Vertical
                landmarks[362].z - landmarks[263].z   # Profundidade
            ], device=self.device)
            
            # Média dos dois olhos
            gaze = (left_eye + right_eye) / 2
            
            # Normaliza
            gaze = (gaze - gaze.mean()) / (gaze.std() + 1e-6)
            
            # Adiciona dimensão de batch
            return gaze.unsqueeze(0)  # [1, 3]
        except Exception as e:
            print(f"Erro na análise do olhar: {e}")
            return torch.zeros(1, 3, device=self.device)
    
    def _analyze_muscle_tension(self, frame: np.ndarray) -> torch.Tensor:
        """Analisa a tensão muscular facial"""
        try:
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return torch.zeros(1, 4, device=self.device)  # [batch, features]
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calcula tensão em diferentes regiões
            tension_features = torch.zeros(4, device=self.device)
            
            # 1. Tensão na testa
            forehead_points = [10, 151, 9, 8]
            tension_features[0] = self._calculate_tension([landmarks[i] for i in forehead_points])
            
            # 2. Tensão ao redor dos olhos
            eye_points = [33, 133, 145, 159]
            tension_features[1] = self._calculate_tension([landmarks[i] for i in eye_points])
            
            # 3. Tensão no nariz
            nose_points = [1, 4, 5, 6]
            tension_features[2] = self._calculate_tension([landmarks[i] for i in nose_points])
            
            # 4. Tensão ao redor da boca
            mouth_points = [61, 291, 0, 17]
            tension_features[3] = self._calculate_tension([landmarks[i] for i in mouth_points])
            
            # Normaliza
            tension_features = (tension_features - tension_features.mean()) / (tension_features.std() + 1e-6)
            
            # Adiciona dimensão de batch
            return tension_features.unsqueeze(0)  # [1, 4]
        except Exception as e:
            print(f"Erro na análise de tensão muscular: {e}")
            return torch.zeros(1, 4, device=self.device)
    
    def _analyze_movement(self, frame: np.ndarray) -> torch.Tensor:
        """Analisa padrões de movimento facial"""
        try:
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return torch.zeros(1, 6, device=self.device)  # [batch, features]
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Adiciona ao histórico
            self.face_history.append(landmarks)
            if len(self.face_history) > self.history_size:
                self.face_history.pop(0)
            
            # Calcula movimento em diferentes regiões
            movement_features = torch.zeros(6, device=self.device)
            
            if len(self.face_history) > 1:
                # 1. Movimento da testa
                forehead_movement = self._calculate_movement(self.face_history[-1][10], self.face_history[-2][10])
                movement_features[0] = forehead_movement
                
                # 2. Movimento das sobrancelhas
                brow_movement = self._calculate_movement(self.face_history[-1][105], self.face_history[-2][105])
                movement_features[1] = brow_movement
                
                # 3. Movimento dos olhos
                eye_movement = self._calculate_movement(self.face_history[-1][33], self.face_history[-2][33])
                movement_features[2] = eye_movement
                
                # 4. Movimento do nariz
                nose_movement = self._calculate_movement(self.face_history[-1][1], self.face_history[-2][1])
                movement_features[3] = nose_movement
                
                # 5. Movimento da boca
                mouth_movement = self._calculate_movement(self.face_history[-1][61], self.face_history[-2][61])
                movement_features[4] = mouth_movement
                
                # 6. Movimento geral
                general_movement = self._calculate_movement(self.face_history[-1][0], self.face_history[-2][0])
                movement_features[5] = general_movement
            
            # Normaliza
            movement_features = (movement_features - movement_features.mean()) / (movement_features.std() + 1e-6)
            
            # Adiciona dimensão de batch
            return movement_features.unsqueeze(0)  # [1, 6]
        except Exception as e:
            print(f"Erro na análise de movimento: {e}")
            return torch.zeros(1, 6, device=self.device)
    
    def _calculate_distance(self, p1, p2) -> float:
        """Calcula a distância euclidiana entre dois pontos"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _calculate_movement(self, p1, p2) -> float:
        """Calcula a distância entre dois pontos de landmark"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _calculate_tension(self, points) -> float:
        """Calcula a tensão muscular baseada na distância entre pontos"""
        if len(points) < 2:
            return 0.0
            
        # Calcula a média das distâncias entre pontos consecutivos
        total_distance = 0.0
        for i in range(len(points) - 1):
            total_distance += self._calculate_distance(points[i], points[i+1])
            
        return total_distance / (len(points) - 1)
    
    def _calculate_detection_confidence(self, landmarks) -> float:
        """Calcula a confiança na detecção da face"""
        try:
            # Verifica se os landmarks principais estão presentes
            key_points = [33, 133, 362, 263, 61, 291, 199, 1]  # Olhos, boca, nariz
            present_points = sum(1 for i in key_points if i < len(landmarks))
            return present_points / len(key_points)
        except:
            return 0.0
    
    def _calculate_landmark_quality(self, landmarks) -> float:
        """Calcula a qualidade dos landmarks"""
        try:
            # Verifica a consistência das distâncias entre landmarks
            distances = []
            for i in range(len(landmarks) - 1):
                dist = self._calculate_distance(landmarks[i], landmarks[i + 1])
                distances.append(dist)
            
            # Calcula o coeficiente de variação
            distances = np.array(distances)
            cv = np.std(distances) / (np.mean(distances) + 1e-6)
            return 1.0 - min(cv, 1.0)  # Quanto menor o CV, melhor a qualidade
        except:
            return 0.0
    
    def _calculate_expression_quality(self, landmarks) -> float:
        """Calcula a qualidade da detecção de expressões"""
        try:
            # Verifica a simetria facial
            left_eye = landmarks[33].y - landmarks[133].y
            right_eye = landmarks[362].y - landmarks[263].y
            eye_symmetry = 1.0 - abs(left_eye - right_eye)
            
            # Verifica a abertura da boca
            mouth_open = landmarks[61].y - landmarks[291].y
            mouth_quality = 1.0 - abs(mouth_open - 0.1)  # Abertura ideal ~0.1
            
            return (eye_symmetry + mouth_quality) / 2
        except:
            return 0.0
    
    def _calculate_movement_quality(self, landmarks) -> float:
        """Calcula a qualidade da detecção de movimentos"""
        try:
            if len(self.face_history) < 2:
                return 0.0
                
            # Calcula a variação dos landmarks entre frames
            prev_landmarks = self.face_history[-2]
            curr_landmarks = self.face_history[-1]
            
            movement = 0.0
            for i in range(min(len(prev_landmarks), len(curr_landmarks))):
                movement += self._calculate_distance(prev_landmarks[i], curr_landmarks[i])
            
            # Normaliza o movimento
            movement = min(movement, 1.0)
            return 1.0 - movement  # Quanto menor o movimento, melhor a qualidade
        except:
            return 0.0
    
    def _get_default_analysis(self, speaker_id: str) -> FaceAnalysis:
        """Retorna uma análise padrão para quando não detecta face"""
        return FaceAnalysis(
            speaker_id=speaker_id,
            emotion_probs=torch.ones(1, 7, device=self.device) / 7,
            micro_expressions=torch.zeros(1, 5, device=self.device),
            gaze_direction=torch.zeros(1, 3, device=self.device),
            muscle_tension=torch.zeros(1, 4, device=self.device),
            movement_patterns=torch.zeros(1, 6, device=self.device),
            face_position={"x": 0, "y": 0, "w": 0, "h": 0},
            detection_confidence=0.0,
            landmark_quality=0.0,
            expression_quality=0.0,
            movement_quality=0.0
        ) 