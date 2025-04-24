import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """Classe base para normalização de features"""
    
    def __init__(self, target_dim: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.target_dim = target_dim
        self.device = device
        self.layer_norm = nn.LayerNorm(target_dim).to(device)
        
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza o tensor para a dimensão alvo"""
        raise NotImplementedError

class AudioFeatureNormalizer(FeatureNormalizer):
    """Normalizador para features de áudio"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # 8 emoções + pitch + intensity + timbre + speech_rate + rhythm + 4 audio_quality
        super().__init__(8 + 1 + 1 + 13 + 1 + 3 + 4, device)
        
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza features de áudio para dimensão padrão"""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            
        # Se o tensor já tem a dimensão correta, apenas normaliza
        if tensor.shape[1] == self.target_dim:
            return self.layer_norm(tensor)
            
        # Caso contrário, projeta para a dimensão correta
        if tensor.shape[1] < self.target_dim:
            # Preenche com zeros
            padding = torch.zeros(tensor.shape[0], self.target_dim - tensor.shape[1], device=self.device)
            tensor = torch.cat([tensor, padding], dim=1)
        else:
            # Reduz dimensão mantendo as features mais importantes
            tensor = tensor[:, :self.target_dim]
            
        return self.layer_norm(tensor)

class FaceFeatureNormalizer(FeatureNormalizer):
    """Normalizador para features faciais"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # 7 emoções + 5 micro_expressions + 3 gaze_direction + 4 muscle_tension + 4 movement_patterns + 4 face_position
        super().__init__(7 + 5 + 3 + 4 + 4 + 4, device)
        
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza features faciais para dimensão padrão"""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            
        # Se o tensor já tem a dimensão correta, apenas normaliza
        if tensor.shape[1] == self.target_dim:
            return self.layer_norm(tensor)
            
        # Caso contrário, projeta para a dimensão correta
        if tensor.shape[1] < self.target_dim:
            # Preenche com zeros
            padding = torch.zeros(tensor.shape[0], self.target_dim - tensor.shape[1], device=self.device)
            tensor = torch.cat([tensor, padding], dim=1)
        else:
            # Reduz dimensão mantendo as features mais importantes
            tensor = tensor[:, :self.target_dim]
            
        return self.layer_norm(tensor)

class TextFeatureNormalizer(FeatureNormalizer):
    """Normalizador para features de texto"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # 7 emoções + sarcasm + humor + polarity + intensity + context_embedding + 4 text_quality
        super().__init__(7 + 1 + 1 + 1 + 1 + 768 + 4, device)
        
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza features de texto para dimensão padrão"""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            
        # Se o tensor já tem a dimensão correta, apenas normaliza
        if tensor.shape[1] == self.target_dim:
            return self.layer_norm(tensor)
            
        # Caso contrário, projeta para a dimensão correta
        if tensor.shape[1] < self.target_dim:
            # Preenche com zeros
            padding = torch.zeros(tensor.shape[0], self.target_dim - tensor.shape[1], device=self.device)
            tensor = torch.cat([tensor, padding], dim=1)
        else:
            # Reduz dimensão mantendo as features mais importantes
            tensor = tensor[:, :self.target_dim]
            
        return self.layer_norm(tensor) 