import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.structures.analysis import AudioAnalysis, TextAnalysis, FaceAnalysis

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFusionModel(nn.Module):
    def __init__(
        self,
        # Face: 7 emoções + 5 micro_expressions + 3 gaze_direction + 4 muscle_tension + 4 movement_patterns + 4 face_position
        face_dim: int = 7 + 5 + 3 + 4 + 4 + 4,
        # Audio: 8 emoções + pitch + intensity + timbre + speech_rate + rhythm + 4 audio_quality
        audio_dim: int = 8 + 1 + 1 + 13 + 1 + 3 + 4,
        # Text: 7 emoções + sarcasm + humor + polarity + intensity + context_embedding + 4 text_quality
        text_dim: int = 7 + 1 + 1 + 1 + 1 + 768 + 4,
        hidden_dim: int = 1024,
        output_dim: int = 7,
        dropout: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.dropout = dropout
        
        # Dimensões fixas
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.face_dim = face_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Camadas de normalização para cada modalidade
        self.audio_norm = nn.LayerNorm(audio_dim).to(device)
        self.text_norm = nn.LayerNorm(text_dim).to(device)
        self.face_norm = nn.LayerNorm(face_dim).to(device)
        
        # Camadas de projeção para cada modalidade
        self.audio_proj = nn.Linear(audio_dim, hidden_dim).to(device)
        self.text_proj = nn.Linear(text_dim, hidden_dim).to(device)
        self.face_proj = nn.Linear(face_dim, hidden_dim).to(device)
        
        # Camadas de processamento para cada modalidade
        self.audio_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(device)
        
        self.text_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(device)
        
        self.face_processor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(device)
        
        # Camadas de fusão
        self.fusion = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        ).to(device)
        
        # Camada de fusão para 2 modalidades
        self.fusion2 = nn.Linear((hidden_dim // 2) * 2, hidden_dim).to(device)
        
        # Pesos iniciais para cada modalidade
        self.audio_weight = nn.Parameter(torch.tensor(0.3, device=device))
        self.text_weight = nn.Parameter(torch.tensor(0.3, device=device))
        self.face_weight = nn.Parameter(torch.tensor(0.4, device=device))
        
        # Normalização dos pesos
        self.softmax = nn.Softmax(dim=0)
        
        # Inicializa os pesos
        self._init_weights()
        
        # Garante que todo o módulo e submódulos sejam movidos para o device
        self.to(device)
    
    def _init_weights(self):
        """Inicializa os pesos das camadas"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _validate_input_dims(self, face_probs: torch.Tensor, audio_probs: torch.Tensor, text_probs: torch.Tensor):
        """Valida as dimensões dos tensores de entrada"""
        if face_probs.shape[-1] != self.face_dim:
            raise ValueError(f"Dimensão incorreta para face_probs. Esperado: {self.face_dim}, Recebido: {face_probs.shape[-1]}")
        if audio_probs.shape[-1] != self.audio_dim:
            raise ValueError(f"Dimensão incorreta para audio_probs. Esperado: {self.audio_dim}, Recebido: {audio_probs.shape[-1]}")
        if text_probs.shape[-1] != self.text_dim:
            raise ValueError(f"Dimensão incorreta para text_probs. Esperado: {self.text_dim}, Recebido: {text_probs.shape[-1]}")
    
    def forward(
        self,
        face_probs: Optional[torch.Tensor] = None,
        audio_probs: Optional[torch.Tensor] = None,
        text_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass do modelo de fusão"""
        try:
            # Verifica quais modalidades estão disponíveis
            available_modalities = []
            if face_probs is not None:
                available_modalities.append("face")
            if audio_probs is not None:
                available_modalities.append("audio")
            if text_probs is not None:
                available_modalities.append("text")

            if not available_modalities:
                raise ValueError("Nenhuma modalidade disponível para fusão")

            # Log das modalidades disponíveis
            logger.debug(f"Modalidades disponíveis: {available_modalities}")

            # Se tivermos apenas uma modalidade, retorna ela
            if len(available_modalities) == 1:
                modality = available_modalities[0]
                if modality == "face":
                    return {"face": face_probs}
                elif modality == "audio":
                    return {"audio": audio_probs}
                else:
                    return {"text": text_probs}

            # Se tivermos duas modalidades, faz fusão parcial
            if len(available_modalities) == 2:
                if "face" in available_modalities and "audio" in available_modalities:
                    logger.debug("Fusão face + áudio")
                    return self._fuse_face_audio(face_probs, audio_probs)
                elif "face" in available_modalities and "text" in available_modalities:
                    logger.debug("Fusão face + texto")
                    return self._fuse_face_text(face_probs, text_probs)
                else:
                    logger.debug("Fusão áudio + texto")
                    return self._fuse_audio_text(audio_probs, text_probs)

            # Se tivermos todas as modalidades, faz fusão completa
            logger.debug("Fusão completa face + áudio + texto")
            return self._fuse_all(face_probs, audio_probs, text_probs)

        except Exception as e:
            logger.error(f"Erro no forward do FusionModel: {str(e)}", exc_info=True)
            # Em caso de erro, retorna a modalidade mais confiável
            if face_probs is not None:
                return {"face": face_probs}
            elif audio_probs is not None:
                return {"audio": audio_probs}
            elif text_probs is not None:
                return {"text": text_probs}
            else:
                raise ValueError("Nenhuma modalidade disponível para retorno de fallback")
    
    def get_weights(self) -> Dict[str, float]:
        """Retorna os pesos atuais de cada modalidade"""
        weights = self.softmax(torch.stack([
            self.audio_weight,
            self.text_weight,
            self.face_weight
        ]))
        return {
            "audio": weights[0].item(),
            "text": weights[1].item(),
            "face": weights[2].item()
        }

    def compute_loss(
        self, 
        face_vec: torch.Tensor, 
        audio_vec: torch.Tensor, 
        text_vec: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computa a perda KL-Divergence entre a predição e o target.
        O target é calculado como média ponderada dos vetores unimodais.
        """
        # Concatena os vetores unimodais
        x = torch.cat([face_vec, audio_vec, text_vec], dim=-1)
        
        # Faz a predição
        pred = self(x)
        
        # Calcula o target ponderado
        weighted_target = (
            self.weights[0] * face_vec + 
            self.weights[1] * audio_vec + 
            self.weights[2] * text_vec
        )
        weighted_target = weighted_target / weighted_target.sum(dim=-1, keepdim=True)
        
        # Computa a perda KL-Divergence
        loss = F.kl_div(
            torch.log(pred + 1e-8), 
            weighted_target, 
            reduction='batchmean'
        )
        
        return loss, pred

    def save(self, path: str):
        """Salva o modelo e seus pesos"""
        # Cria o diretório se não existir
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Obtém os pesos atuais
        weights = self.get_weights()
        
        # Salva o estado do modelo e as dimensões
        torch.save({
            'model_state_dict': self.state_dict(),
            'weights': weights,
            'audio_dim': self.audio_dim,
            'text_dim': self.text_dim,
            'face_dim': self.face_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'AdvancedFusionModel':
        """Carrega o modelo e seus pesos"""
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Cria novo modelo com as dimensões salvas
            model = cls(
                audio_dim=checkpoint['audio_dim'],
                text_dim=checkpoint['text_dim'],
                face_dim=checkpoint['face_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                output_dim=checkpoint['output_dim'],
                dropout=checkpoint['dropout'],
                device=device
            )
            
            # Carrega os pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Atualiza os pesos de fusão
            weights = checkpoint.get('weights', {'audio': 0.3, 'text': 0.3, 'face': 0.4})
            model.audio_weight.data.fill_(weights['audio'])
            model.text_weight.data.fill_(weights['text'])
            model.face_weight.data.fill_(weights['face'])
            
            return model
        except FileNotFoundError:
            logger.warning(f"Checkpoint não encontrado em {path}. Criando novo modelo...")
            # Cria o diretório se não existir
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Cria novo modelo com dimensões padrão
            model = cls(device=device)
            # Salva o modelo
            model.save(path)
            return model

    def _fuse_face_audio(self, face_probs: torch.Tensor, audio_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fusão de face e áudio"""
        try:
            # Normaliza os inputs
            face_norm = self.face_norm(face_probs)
            audio_norm = self.audio_norm(audio_probs)
            
            # Projeta para dimensão oculta
            face_hidden = self.face_proj(face_norm)
            audio_hidden = self.audio_proj(audio_norm)
            
            # Processa cada modalidade
            face_processed = self.face_processor(face_hidden)
            audio_processed = self.audio_processor(audio_hidden)
            
            # Concatena e faz fusão
            fused = torch.cat([face_processed, audio_processed], dim=-1)  # → [batch, 1024]
            
            # Usa fusion2 para ajustar a dimensão
            x = self.fusion2(fused)  # agora aceita 1024 → hidden_dim
            
            # Aplica as camadas restantes do fusion
            for layer in list(self.fusion.children())[1:]:
                x = layer(x)
            
            return {"face": face_probs, "audio": audio_probs, "fused": x}
        except Exception as e:
            logger.error(f"Erro na fusão face + áudio: {str(e)}", exc_info=True)
            # Em caso de erro, retorna a modalidade mais confiável
            if face_probs is not None:
                return {"face": face_probs}
            else:
                return {"audio": audio_probs}

    def _fuse_face_text(self, face_probs: torch.Tensor, text_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fusão de face e texto"""
        try:
            # Normaliza os inputs
            face_norm = self.face_norm(face_probs)
            text_norm = self.text_norm(text_probs)
            
            # Projeta para dimensão oculta
            face_hidden = self.face_proj(face_norm)
            text_hidden = self.text_proj(text_norm)
            
            # Processa cada modalidade
            face_processed = self.face_processor(face_hidden)
            text_processed = self.text_processor(text_hidden)
            
            # Concatena e faz fusão
            fused = torch.cat([face_processed, text_processed], dim=-1)
            fused = self.fusion(fused)
            
            return {"face": face_probs, "text": text_probs, "fused": fused}
        except Exception as e:
            logger.error(f"Erro na fusão face + texto: {str(e)}", exc_info=True)
            # Em caso de erro, retorna a modalidade mais confiável
            if face_probs is not None:
                return {"face": face_probs}
            else:
                return {"text": text_probs}

    def _fuse_audio_text(self, audio_probs: torch.Tensor, text_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fusão de áudio e texto"""
        try:
            # Normaliza os inputs
            audio_norm = self.audio_norm(audio_probs)
            text_norm = self.text_norm(text_probs)
            
            # Projeta para dimensão oculta
            audio_hidden = self.audio_proj(audio_norm)
            text_hidden = self.text_proj(text_norm)
            
            # Processa cada modalidade
            audio_processed = self.audio_processor(audio_hidden)
            text_processed = self.text_processor(text_hidden)
            
            # Concatena e faz fusão
            fused = torch.cat([audio_processed, text_processed], dim=-1)
            fused = self.fusion(fused)
            
            return {"audio": audio_probs, "text": text_probs, "fused": fused}
        except Exception as e:
            logger.error(f"Erro na fusão áudio + texto: {str(e)}", exc_info=True)
            # Em caso de erro, retorna a modalidade mais confiável
            if audio_probs is not None:
                return {"audio": audio_probs}
            else:
                return {"text": text_probs}

    def _fuse_all(self, face_probs: torch.Tensor, audio_probs: torch.Tensor, text_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fusão completa de todas as modalidades"""
        try:
            # Normaliza os inputs
            face_norm = self.face_norm(face_probs)
            audio_norm = self.audio_norm(audio_probs)
            text_norm = self.text_norm(text_probs)
            
            # Projeta para dimensão oculta
            face_hidden = self.face_proj(face_norm)
            audio_hidden = self.audio_proj(audio_norm)
            text_hidden = self.text_proj(text_norm)
            
            # Processa cada modalidade
            face_processed = self.face_processor(face_hidden)
            audio_processed = self.audio_processor(audio_hidden)
            text_processed = self.text_processor(text_hidden)
            
            # Concatena e faz fusão
            fused = torch.cat([face_processed, audio_processed, text_processed], dim=-1)
            fused = self.fusion(fused)
            
            return {"face": face_probs, "audio": audio_probs, "text": text_probs, "fused": fused}
        except Exception as e:
            logger.error(f"Erro na fusão completa: {str(e)}", exc_info=True)
            # Em caso de erro, retorna a modalidade mais confiável
            if face_probs is not None:
                return {"face": face_probs}
            elif audio_probs is not None:
                return {"audio": audio_probs}
            else:
                return {"text": text_probs}

# Alias para compatibilidade
FusionModel = AdvancedFusionModel 