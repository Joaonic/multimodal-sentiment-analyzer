import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from tqdm import tqdm

from src.processors.offline_processor import OfflineProcessor
from src.structures import (
    ModelConfig,
    ProcessingConfig,
    StreamingConfig,
    AnalysisResult,
    FaceEmotionVector,
    AudioEmotionVector,
    TextEmotionVector
)

# Configuração do logger
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model_config: ModelConfig,
        processing_config: ProcessingConfig,
        hf_token: Optional[str] = None
    ):
        self.model_config = model_config
        self.processing_config = processing_config
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Inicializa o processador
        self.processor = OfflineProcessor(
            model_config=asdict(model_config),
            processing_config=asdict(processing_config),
            hf_token=self.hf_token
        )
        
        # Lista de emoções
        self.emotions = [
            "feliz", "triste", "raiva", "medo",
            "surpresa", "nojo", "neutro"
        ]
        
        # Dicionário para armazenar resultados
        self.results = {
            "face": {"true": [], "pred": []},
            "audio": {"true": [], "pred": []},
            "text": {"true": [], "pred": []},
            "fused": {"true": [], "pred": []}
        }
    
    def evaluate_video(
        self,
        video_path: str,
        ground_truth: Dict[str, List[str]],
        output_dir: str = "evaluation"
    ) -> Dict[str, Dict[str, float]]:
        """Avalia o modelo em um vídeo com ground truth"""
        # Cria diretório de saída
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Processa o vídeo
        results = []
        
        def on_result(result: AnalysisResult):
            results.append(result)
        
        def on_error(error: Exception):
            logger.error(f"Erro durante processamento: {str(error)}")
        
        def on_progress(progress: float):
            logger.info(f"Progresso: {progress:.1%}")
        
        self.processor.process_video(
            video_path=video_path,
            on_result=on_result,
            on_error=on_error,
            on_progress=on_progress
        )
        
        # Compara com ground truth
        metrics = {}
        for modality in ["face", "audio", "text", "fused"]:
            metrics[modality] = self._calculate_metrics(
                results=results,
                ground_truth=ground_truth,
                modality=modality
            )
        
        # Gera visualizações
        self._generate_visualizations(
            results=results,
            ground_truth=ground_truth,
            output_path=output_path
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        results: List[AnalysisResult],
        ground_truth: Dict[str, List[str]],
        modality: str
    ) -> Dict[str, float]:
        """Calcula métricas para uma modalidade específica"""
        # Extrai predições e ground truth
        y_true = []
        y_pred = []
        
        for result in results:
            # Ground truth para o segmento
            segment_truth = ground_truth.get(
                f"{result.start_time:.1f}-{result.end_time:.1f}",
                ["neutro"]  # Default se não houver anotação
            )
            
            # Predição do modelo
            if modality == "face":
                pred = result.face_analysis
            elif modality == "audio":
                pred = result.audio_analysis
            elif modality == "text":
                pred = result.text_analysis
            else:  # fused
                pred = result.fused_analysis
            
            # Adiciona à lista
            y_true.extend(segment_truth)
            y_pred.extend([pred.dominant_emotion] * len(segment_truth))
        
        # Calcula métricas
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self.emotions,
                output_dict=True
            )
        }
        
        # Calcula ROC-AUC para cada emoção
        for emotion in self.emotions:
            y_true_binary = [1 if e == emotion else 0 for e in y_true]
            y_pred_binary = [1 if e == emotion else 0 for e in y_pred]
            
            try:
                metrics[f"roc_auc_{emotion}"] = roc_auc_score(
                    y_true_binary, y_pred_binary
                )
            except ValueError:
                metrics[f"roc_auc_{emotion}"] = 0.0
        
        return metrics
    
    def _generate_visualizations(
        self,
        results: List[AnalysisResult],
        ground_truth: Dict[str, List[str]],
        output_path: Path
    ):
        """Gera visualizações dos resultados"""
        # Matriz de confusão para cada modalidade
        for modality in ["face", "audio", "text", "fused"]:
            y_true = []
            y_pred = []
            
            for result in results:
                # Ground truth
                segment_truth = ground_truth.get(
                    f"{result.start_time:.1f}-{result.end_time:.1f}",
                    ["neutro"]
                )
                
                # Predição
                if modality == "face":
                    pred = result.face_analysis
                elif modality == "audio":
                    pred = result.audio_analysis
                elif modality == "text":
                    pred = result.text_analysis
                else:  # fused
                    pred = result.fused_analysis
                
                y_true.extend(segment_truth)
                y_pred.extend([pred.dominant_emotion] * len(segment_truth))
            
            # Plota matriz de confusão
            cm = confusion_matrix(y_true, y_pred, labels=self.emotions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.emotions,
                yticklabels=self.emotions
            )
            plt.title(f"Matriz de Confusão - {modality}")
            plt.xlabel("Predição")
            plt.ylabel("Ground Truth")
            plt.savefig(output_path / f"confusion_matrix_{modality}.png")
            plt.close()
        
        # Timeline de emoções
        plt.figure(figsize=(15, 5))
        for modality in ["face", "audio", "text", "fused"]:
            times = []
            emotions = []
            
            for result in results:
                if modality == "face":
                    pred = result.face_analysis
                elif modality == "audio":
                    pred = result.audio_analysis
                elif modality == "text":
                    pred = result.text_analysis
                else:  # fused
                    pred = result.fused_analysis
                
                times.append(result.start_time)
                emotions.append(self.emotions.index(pred.dominant_emotion))
            
            plt.plot(times, emotions, label=modality)
        
        plt.yticks(range(len(self.emotions)), self.emotions)
        plt.title("Timeline de Emoções")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Emoção")
        plt.legend()
        plt.savefig(output_path / "emotion_timeline.png")
        plt.close()

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Configurações
    model_config = ModelConfig(
        device=os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        face_model_name="deepface",
        audio_model_name="speechbrain/emotion-recognition-wav2vec2-iemocap",
        text_model_name="neuralmind/bert-base-portuguese-cased",
        fusion_model_path="checkpoints/fusion_model.pth",
        weights={
            "face": 0.4,
            "audio": 0.3,
            "text": 0.3
        }
    )
    
    processing_config = ProcessingConfig(
        segment_duration=5.0,
        min_speech_duration=0.5,
        min_pause_duration=0.3,
        output_dir="output",
        temp_dir="temp"
    )
    
    # Cria avaliador
    evaluator = ModelEvaluator(
        model_config=model_config,
        processing_config=processing_config
    )
    
    # Exemplo de ground truth
    ground_truth = {
        "0.0-5.0": ["feliz", "feliz", "neutro"],
        "5.0-10.0": ["triste", "triste", "neutro"],
        "10.0-15.0": ["raiva", "raiva", "neutro"]
    }
    
    # Avalia o modelo
    metrics = evaluator.evaluate_video(
        video_path="teste.mp4",
        ground_truth=ground_truth,
        output_dir="evaluation"
    )
    
    # Salva métricas
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Avaliação concluída. Métricas salvas em evaluation/metrics.json")

if __name__ == "__main__":
    main() 