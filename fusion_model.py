import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class FusionModel(nn.Module):
    """
    Rede neural para fusão de vetores unimodais de emoção.
    Entrada: vetor de dimensão 21 (3 x 7 emoções).
    Saída: distribuição de probabilidades sobre 7 emoções.
    """
    def __init__(
        self, 
        input_dim: int = 21, 
        hidden_dim: int = 256, 
        output_dim: int = 7, 
        dropout: float = 0.5,
        weights: Optional[torch.Tensor] = None
    ):
        super(FusionModel, self).__init__()
        
        # Pesos para cada modalidade (face, áudio, texto)
        self.weights = weights if weights is not None else torch.tensor([0.4, 0.3, 0.3])
        
        # Camadas do modelo
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Regularização
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)
        
        # Inicialização dos pesos
        self._init_weights()

    def _init_weights(self):
        """Inicialização dos pesos das camadas lineares"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - ReLU após cada camada oculta
        - LayerNorm para normalização
        - Dropout para regularização
        - Softmax na saída para probabilidades
        """
        # Garante que o input tenha dimensão de batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Camada oculta 1 + normalização + ativação + dropout
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        # Camada oculta 2 + normalização + ativação + dropout
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        # Camada de saída + softmax
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

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
        torch.save({
            'model_state_dict': self.state_dict(),
            'weights': self.weights,
            'input_dim': self.fc1.in_features,
            'hidden_dim': self.fc1.out_features,
            'output_dim': self.fc3.out_features,
            'dropout': self.dropout.p
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'FusionModel':
        """Carrega o modelo e seus pesos"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            dropout=checkpoint['dropout'],
            weights=checkpoint['weights']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 