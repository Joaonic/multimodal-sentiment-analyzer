import torch
import numpy as np
from fusion_model import FusionModel
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalAnalyzer:
    """
    Classe para análise multimodal de sentimentos.
    Suporta análise de vídeo completo e streaming.
    """
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        emotion_labels: List[str] = None
    ):
        """
        Inicializa o analisador multimodal.
        
        Args:
            model_path: Caminho para o modelo treinado
            device: Dispositivo para execução ('cuda' ou 'cpu')
            emotion_labels: Lista de labels de emoção
        """
        # Configura dispositivo
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Carrega modelo
        self.model = FusionModel.load(model_path, self.device)
        self.model.eval()
        
        # Configura labels de emoção
        self.emotion_labels = emotion_labels or [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        
        logger.info(f"Modelo carregado em {self.device}")

    def analyze_segment(
        self,
        face_vec: np.ndarray,
        audio_vec: np.ndarray,
        text_vec: np.ndarray
    ) -> Dict:
        """
        Analisa um segmento de dados multimodais.
        
        Args:
            face_vec: Vetor de emoções faciais (7-dim)
            audio_vec: Vetor de emoções do áudio (7-dim)
            text_vec: Vetor de emoções do texto (7-dim)
            
        Returns:
            Dict com resultados da análise
        """
        # Converte para tensores
        face_tensor = torch.tensor(face_vec, dtype=torch.float32).to(self.device)
        audio_tensor = torch.tensor(audio_vec, dtype=torch.float32).to(self.device)
        text_tensor = torch.tensor(text_vec, dtype=torch.float32).to(self.device)
        
        # Faz predição
        with torch.no_grad():
            _, pred = self.model.compute_loss(face_tensor, audio_tensor, text_tensor, None)
            pred = pred.cpu().numpy().squeeze()
            
        # Obtém emoção dominante
        dominant_idx = np.argmax(pred)
        dominant_emotion = self.emotion_labels[dominant_idx]
        
        return {
            'emotions': dict(zip(self.emotion_labels, pred.tolist())),
            'dominant_emotion': dominant_emotion,
            'confidence': float(pred[dominant_idx])
        }

    def analyze_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Analisa um vídeo completo.
        
        Args:
            video_path: Caminho do vídeo
            output_path: Caminho para salvar resultados (opcional)
            
        Returns:
            Dict com resultados da análise
        """
        # TODO: Implementar extração de frames, áudio e texto do vídeo
        # Por enquanto, apenas um placeholder
        results = {
            'video_path': video_path,
            'segments': []
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

    def analyze_streaming(
        self,
        duration: float = 5.0,
        callback: Optional[callable] = None
    ) -> None:
        """
        Analisa streaming de vídeo/áudio em tempo real.
        
        Args:
            duration: Duração de cada segmento em segundos
            callback: Função para processar resultados (opcional)
        """
        # TODO: Implementar captura e análise em tempo real
        # Por enquanto, apenas um placeholder
        logger.info(f"Iniciando análise de streaming com duração de segmento: {duration}s")
        
        if callback:
            # Exemplo de callback
            results = {
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 1.0
            }
            callback(results)

def main():
    # Exemplo de uso
    model_path = 'checkpoints/best_model.pt'
    analyzer = MultimodalAnalyzer(model_path)
    
    # Exemplo de análise de segmento
    face_vec = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])  # Exemplo
    audio_vec = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3])  # Exemplo
    text_vec = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3])  # Exemplo
    
    result = analyzer.analyze_segment(face_vec, audio_vec, text_vec)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main() 