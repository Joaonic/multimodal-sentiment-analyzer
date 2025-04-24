from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega variáveis do .env
load_dotenv()

# Verifica token do HuggingFace
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    logger.error("Token do HuggingFace não encontrado no arquivo .env")
else:
    logger.info(f"Token do HuggingFace carregado: {hf_token[:5]}...{hf_token[-5:]}")

# Diretórios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Configurações dos modelos
MODEL_CONFIG = {
    'device': os.getenv('MODEL_DEVICE', 'cuda'),  # ou 'cpu'
    'face_model': 'opencv',
    'audio_model': 'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
    'text_model': 'neuralmind/bert-base-portuguese-cased',
    'fusion_model': str(CHECKPOINTS_DIR / "best_model.pt"),
    'weights': (0.4, 0.3, 0.3),  # face, audio, text
    'hf_token': hf_token  # Token do HuggingFace
}

# Configurações de processamento
PROCESSING_CONFIG = {
    'segment_duration': 5.0,
    'min_speech_duration': 0.5,
    'min_pause_duration': 0.5,
    'output_dir': str(OUTPUT_DIR),
    'temp_dir': str(TEMP_DIR)
}

# Configurações de streaming
STREAMING_CONFIG = {
    'video_source': 0,  # Webcam
    'audio_source': 0,  # Microfone padrão
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s'),
    'filename': str(OUTPUT_DIR / 'analysis.log')
}

# Configurações de diarização
DIARIZATION_CONFIG = {
    'model': 'pyannote/speaker-diarization',
    'min_speakers': 1,
    'max_speakers': 4,
    'hf_token': hf_token  # Token do HuggingFace
}

# Configurações de transcrição
TRANSCRIPTION_CONFIG = {
    'model': 'openai/whisper-medium',
    'language': 'pt',
    'task': 'transcribe'
}

# Configurações de análise facial
FACE_ANALYSIS_CONFIG = {
    'backend': 'opencv',
    'actions': ['emotion'],
    'enforce_detection': False,
    'align': True
}

# Configurações de análise de áudio
AUDIO_ANALYSIS_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'format': 'wav'
}

# Configurações de análise de texto
TEXT_ANALYSIS_CONFIG = {
    'max_length': 512,
    'truncation': True,
    'padding': True
}

# Cria diretórios necessários
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 