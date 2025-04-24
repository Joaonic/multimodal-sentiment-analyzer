import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

from src.config.logging_config import setup_logging

# Configuração do logger
logger = logging.getLogger(__name__)

from src.structures import (
    ModelConfig,
    ProcessingConfig,
    StreamingConfig,
    AnalysisResult
)
from src.processors.offline_processor import OfflineProcessor
from src.processors.streaming_processor import StreamingProcessor
from src.utils import create_directories

def main():
    # Configura o logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando aplicação")
    
    try:
        # Carrega variáveis de ambiente
        load_dotenv()
        
        # Configuração do sistema
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        create_directories()
        
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
        
        streaming_config = StreamingConfig(
            video_source=0,
            audio_source=None,
            sample_rate=16000,
            channels=1,
            chunk_size=1024
        )
        
        # Callbacks para processamento
        def on_result(result: AnalysisResult):
            """Callback chamado quando um resultado de análise está pronto"""
            logger.info(f"Resultado recebido: {result.dominant_emotion} (confiança: {result.confidence:.2f})")
            
            # Salva resultado em JSON
            output_file = Path(processing_config.output_dir) / "results.json"
            with open(output_file, "a") as f:
                json.dump(asdict(result), f)
                f.write("\n")
        
        def on_error(error: Exception):
            """Callback chamado quando ocorre um erro"""
            logger.error(f"Erro durante processamento: {str(error)}")
        
        def on_progress(progress: float):
            """Callback chamado para atualizar progresso"""
            logger.info(f"Progresso: {progress:.1%}")
        
        # Processamento offline
        def process_video(video_path: str, hf_token: Optional[str] = None):
            """Processa um vídeo offline"""
            processor = OfflineProcessor(
                model_config=asdict(model_config),
                processing_config=asdict(processing_config),
                hf_token=hf_token or os.getenv("HF_TOKEN")
            )
            
            processor.process_video(
                video_path=video_path,
                on_result=on_result,
                on_error=on_error,
                on_progress=on_progress
            )
        
        # Processamento em streaming
        def process_streaming(duration: float = 5.0):
            """Processa streaming em tempo real"""
            processor = StreamingProcessor(
                model_config=asdict(model_config),
                streaming_config=asdict(streaming_config)
            )
            
            processor.run(
                duration=duration,
                callback=on_result
            )
        
        # Exemplo de uso
        if __name__ == "__main__":
            import argparse
            
            parser = argparse.ArgumentParser(description="Análise de Sentimentos Multimodal")
            parser.add_argument("--mode", choices=["offline", "streaming"], required=True,
                              help="Modo de operação: offline ou streaming")
            parser.add_argument("--video", help="Caminho do vídeo para processamento offline")
            parser.add_argument("--duration", type=float, default=5.0,
                              help="Duração do processamento em streaming (em segundos)")
            parser.add_argument("--hf-token", help="Token do HuggingFace para diarização")
            
            args = parser.parse_args()
            
            if args.mode == "offline":
                if not args.video:
                    parser.error("--video é obrigatório no modo offline")
                process_video(args.video, args.hf_token)
            else:
                process_streaming(args.duration)
    except Exception as e:
        logger.error(f"Erro na aplicação: {str(e)}", exc_info=True)
    finally:
        logger.info(f"Aplicação encerrada. Logs salvos em: {log_file}")

if __name__ == "__main__":
    main() 