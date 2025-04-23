import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from fusion_model import FusionModel
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, List, Tuple
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AMIDataset(Dataset):
    """
    Dataset para o AMI Corpus.
    Carrega vetores de emoção pré-processados para face, áudio e texto.
    """
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = data_dir
        self.split = split
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Carrega os dados do diretório especificado"""
        data = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        for file in os.listdir(split_dir):
            if file.endswith('.json'):
                with open(os.path.join(split_dir, file), 'r') as f:
                    data.extend(json.load(f))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        
        # Carrega os vetores de emoção
        face_vec = torch.tensor(item['face_vec'], dtype=torch.float32)
        audio_vec = torch.tensor(item['audio_vec'], dtype=torch.float32)
        text_vec = torch.tensor(item['text_vec'], dtype=torch.float32)
        target = torch.tensor(item['target'], dtype=torch.float32)
        
        return face_vec, audio_vec, text_vec, target

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int = 100,
    patience: int = 10,
    checkpoint_dir: str = 'checkpoints'
) -> None:
    """
    Função de treinamento do modelo.
    """
    # Cria diretório para checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Inicializa variáveis para early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Loop de treinamento
    for epoch in range(num_epochs):
        # Modo de treinamento
        model.train()
        train_loss = 0.0
        
        # Barra de progresso
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for face_vec, audio_vec, text_vec, target in pbar:
            # Move dados para o dispositivo
            face_vec = face_vec.to(device)
            audio_vec = audio_vec.to(device)
            text_vec = text_vec.to(device)
            target = target.to(device)
            
            # Forward pass e cálculo da perda
            loss, _ = model.compute_loss(face_vec, audio_vec, text_vec, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Atualiza métricas
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calcula perda média de treinamento
        train_loss /= len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for face_vec, audio_vec, text_vec, target in val_loader:
                face_vec = face_vec.to(device)
                audio_vec = audio_vec.to(device)
                text_vec = text_vec.to(device)
                target = target.to(device)
                
                loss, _ = model.compute_loss(face_vec, audio_vec, text_vec, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Log das métricas
        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Early stopping e salvamento do modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

def main():
    # Configurações
    data_dir = 'data/ami'  # Diretório com os dados do AMI
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100
    patience = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cria datasets
    train_dataset = AMIDataset(data_dir, 'train')
    val_dataset = AMIDataset(data_dir, 'val')
    
    # Cria dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Inicializa modelo
    model = FusionModel().to(device)
    
    # Otimizador
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Treina modelo
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience
    )

if __name__ == '__main__':
    main() 