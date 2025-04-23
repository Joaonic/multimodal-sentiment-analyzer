# %% [markdown]
"""
# Análise de Sentimentos Multimodal com Diarização e Aprendizado Contínuo

Este notebook implementa um pipeline completo que:
- Lê um vídeo ou stream (áudio e vídeo) e extrai frames e áudio.
- Realiza **diarização** do áudio utilizando o *pyannote.audio*.
- Extrai a emoção facial de cada frame com o *DeepFace*.
- Realiza a análise de emoção do áudio com um modelo pré‑treinado do *SpeechBrain*.
- Transcreve o áudio com *Whisper* e classifica a emoção do texto com um modelo do *Transformers*.
- Converte cada uma dessas análises em vetores de probabilidade (7 dimensões: _angry, disgust, fear, happy, sad, surprise, neutral_).
- Usa um **modelo de fusão** (rede neural em PyTorch) que recebe os 3 vetores unimodais (concatenados em um vetor de dimensão 21) e gera uma previsão final. Esse modelo é treinado continuamente de forma auto‑supervisionada, utilizando como pseudo‑label a média dos três vetores.
- Agrega os resultados por locutor e gera uma saída JSON, incluindo "padrões" extraídos (ex.: detecção de consistência de emoção).

O pipeline roda em dois modos:
1. **Offline:** Processa um vídeo gravado.
2. **Streaming:** Captura áudio e vídeo em tempo real (utilizando webcam e microfone via PyAudio) e processa em blocos de tempo.
"""
from speechbrain.inference import foreign_class
from typing import List, Dict, Optional, Union
from analysis_types import EmotionVector, SegmentAnalysis, SpeakerPattern, SpeakerData, AnalysisResult, CompleteAnalysisResult

# %% [markdown]
"""
## Instalação das Dependências

Execute os seguintes comandos para instalar as dependências necessárias:
```bash
pip install opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyaudio
pip install openai-whisper
pip install deepface
pip install pyannote.audio
pip install speechbrain
pip install transformers
pip install ffmpeg-python

# Instalação do FFmpeg (necessário para processamento de áudio/vídeo)
apt-get install ffmpeg
```

Observações importantes:
1. Você precisará reiniciar o runtime após instalar as dependências
2. A instalação do PyAudio pode apresentar problemas no Colab. Se isso acontecer, tente:
   ```bash
   apt-get install portaudio19-dev
   pip install pyaudio
   ```
3. Para usar o modo streaming, você precisará permitir o acesso à webcam
4. Configure seu token do HuggingFace na variável HF_TOKEN
"""
# %% [markdown]
"""
# 0. CONFIGURAÇÕES DE MEMÓRIA CUDA
"""
# %%
import json
import os
import gc
# Enable expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import queue
import subprocess
import time
import wave

# Importação das bibliotecas necessárias
import cv2
import numpy as np
import pyaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import whisper
from deepface import DeepFace
from pyannote.audio import Pipeline
from speechbrain.inference.interfaces import foreign_class
from transformers import pipeline as hf_pipeline
from tqdm import tqdm  # Para barras de progresso
import tensorflow as tf

# Configurações CUDA
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
print(f"PyTorch versão: {torch.__version__}")
print(f"OpenCV versão: {cv2.__version__}")
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
print("Versão do TensorFlow:", tf.__version__)

# Verificação da versão do CuDNN de forma mais robusta
try:
    cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Não detectado')
    print(f"Versão do CuDNN: {cudnn_version}")
except Exception as e:
    print("Versão do CuDNN: Não foi possível detectar a versão do CuDNN")
    print(f"Detalhes do erro: {str(e)}")

try:
    print(f"Whisper versão: {whisper.__version__}")
except AttributeError:
    pass

# Configurações CUDA e tipos de dados
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Força uso da GPU 0
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Habilita XLA
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Otimiza threads GPU
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # Desabilita autotune do cuDNN

# Configurações específicas para RTX 4060 e CUDA 12.4
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Otimiza alocação de memória
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'  # Otimiza batch norm
os.environ['TF_ENABLE_CUDNN_FRONTEND'] = '1'  # Habilita frontend otimizado do cuDNN

# Força versão específica do CuDNN
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH'
os.environ['CUDNN_PATH'] = '/usr/local/cuda-12.4'

# Configurações PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # Habilita TF32 para RTX
torch.backends.cudnn.allow_tf32 = True  # Habilita TF32 para cuDNN

# Configurações CUDA e tipos de dados
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Força uso da GPU 0
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Habilita XLA
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Otimiza threads GPU
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # Desabilita autotune do cuDNN

# %% [markdown]
"""
## 1. Parâmetros e Configurações Globais
"""
# %%
HF_TOKEN = "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"  # Substitua pelo seu token do HuggingFace
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilizando {DEVICE}")

# Carregamento dos modelos com tipos de dados corretos
asr_model = whisper.load_model("medium", device=DEVICE).to(DEVICE).float()  # Usando float32 ao invés de half

# Configuração do SpeechBrain com otimizações
SER_MODEL = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/ser",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": DEVICE},
)

# Inicializa o pipeline de análise de emoção textual
text_emotion_pipeline = hf_pipeline(
    "text-classification",
    model="neuralmind/bert-base-portuguese-cased",
    return_all_scores=True,
    device=0 if DEVICE == "cuda" else -1,
    torch_dtype=torch.float32
)

# Configurações de memória
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
"""
## 2. Funções de Pré-processamento

Funções para extrair o áudio de um vídeo (usando FFmpeg) e para carregar os frames e timestamps.
"""
# %%
def extract_audio(video_path, audio_output=None):
    """
    Extrai o áudio do vídeo utilizando FFmpeg e salva em WAV (16 kHz mono).
    """
    try:
        # Verifica se o arquivo de vídeo existe
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        # Define o caminho do arquivo de áudio
        if audio_output is None:
            # Usa o mesmo diretório do vídeo com nome padrão
            video_dir = os.path.dirname(video_path)
            audio_output = os.path.join(video_dir, "extracted_audio.wav")
        else:
            # Converte para caminho absoluto
            audio_output = os.path.abspath(audio_output)

        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(audio_output), exist_ok=True)

        print(f"Extraindo áudio para: {audio_output}")

        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_output
        ]

        # Executa o comando e captura a saída
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Verifica se o comando foi executado com sucesso
        if result.returncode != 0:
            raise RuntimeError(f"Erro ao extrair áudio: {result.stderr}")

        # Verifica se o arquivo foi criado
        if not os.path.exists(audio_output):
            raise RuntimeError(f"Arquivo de áudio não foi criado: {audio_output}")

        print(f"Áudio extraído com sucesso: {audio_output}")
        return audio_output

    except Exception as e:
        print(f"Erro ao extrair áudio: {str(e)}")
        raise


def load_video_frames(video_path):
    """
    Carrega os frames do vídeo com OpenCV e retorna:
    - frames: lista de imagens (numpy arrays)
    - timestamps: lista de tempos (em segundos) para cada frame
    - fps: frames por segundo do vídeo
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        timestamps.append(frame_index / fps)
        frame_index += 1
    cap.release()
    return frames, timestamps, fps

# %% [markdown]
"""
## 3. Diarização de Áudio

Utiliza o pipeline do *pyannote.audio* para segmentar o áudio por locutor.
"""
# %%
def perform_diarization(audio_path, hf_token):
    """
    Executa a diarização no áudio e retorna uma lista de segmentos:
    Cada segmento é um dicionário com: start, end, speaker.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    pipeline.to(torch.device(DEVICE))
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments

# %% [markdown]
"""
## 4. Análise Unimodal

### 4.1 Análise Facial
Extrai um vetor de probabilidade (7 dimensões) para as emoções usando DeepFace.
### 4.2 Análise de Áudio
Utiliza um modelo pré‑treinado do SpeechBrain para reconhecimento de emoção no áudio.
### 4.3 Análise de Texto
Transcreve o áudio com Whisper e classifica a emoção do texto utilizando um pipeline do Transformers.
"""
# %%
def analyze_face_emotion_vector(frame):
    """
    Analisa a emoção no frame utilizando DeepFace e retorna um vetor de 7 dimensões.
    """
    try:
        # Configuração do DeepFace para usar GPU
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            align=True  # Adiciona alinhamento facial
        )

        if isinstance(result, list):
            result = result[0]

        emotion_dict = result.get("emotion", {})
        face_vec = np.array([
            emotion_dict.get("angry", 0),
            emotion_dict.get("disgust", 0),
            emotion_dict.get("fear", 0),
            emotion_dict.get("happy", 0),
            emotion_dict.get("sad", 0),
            emotion_dict.get("surprise", 0),
            emotion_dict.get("neutral", 0)
        ], dtype=np.float32)

        if np.sum(face_vec) > 0:
            face_vec = face_vec / np.sum(face_vec)
    except Exception as e:
        print("Erro na análise facial:", e)
        face_vec = np.zeros(7, dtype=np.float32)
    return face_vec


def analyze_audio_emotion(audio_segment_path: str) -> np.ndarray:
    """
    Analisa o segmento de áudio e retorna um vetor de probabilidades (7 dims):
    [angry, disgust, fear, happy, sad, surprise, neutral]
    """
    try:
        # classify_file returns: out_prob (tensor of shape [1,4]), score, index, label_name
        out_prob, score, index, label_name = SER_MODEL.classify_file(audio_segment_path)
        probs = out_prob.squeeze().cpu().numpy()  # shape (4,)

        # Map the 4 emotion probs into our 7‐dim vector
        audio_vec = np.zeros(7, dtype=np.float32)
        mapping = {
            "angry": 0,
            "happy": 3,
            "neutral": 6,
            "sad": 4
        }
        labels_4 = ["angry", "happy", "neutral", "sad"]
        
        # Normaliza as probabilidades
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
        
        for i, lbl in enumerate(labels_4):
            audio_vec[mapping[lbl]] = probs[i]

        # Adiciona pequeno ruído para evitar zeros
        audio_vec = audio_vec + 1e-6
        # Renormaliza
        audio_vec = audio_vec / np.sum(audio_vec)

    except Exception as e:
        print(f"Erro na análise de áudio (SER): {e}")
        # Retorna distribuição uniforme em caso de erro
        audio_vec = np.ones(7, dtype=np.float32) / 7

    return audio_vec


def transcribe_audio(audio_segment_path):
    """
    Transcreve o áudio utilizando o modelo Whisper (em português).
    """
    try:
        result = asr_model.transcribe(
            audio_segment_path,
            language="pt",
            fp16=False  # Desabilitando fp16 para evitar problemas de tipo
        )
        return result["text"]
    except Exception as e:
        print("Erro na transcrição:", e)
        return ""


def analyze_text_emotion(text):
    """
    Analisa a emoção do texto e retorna um vetor 7-dim com probabilidades.
    """
    try:
        if not text or len(text.strip()) == 0:
            return np.ones(7, dtype=np.float32) / 7

        # Análise de sentimento em português
        results = text_emotion_pipeline(text)[0]
        text_vec = np.zeros(7, dtype=np.float32)
        
        # Mapeamento das emoções do modelo para nosso vetor
        for item in results:
            if item["label"].lower() == "negative":
                # Distribui entre angry, disgust, fear, sad
                text_vec[0] += item["score"] * 0.3  # angry
                text_vec[1] += item["score"] * 0.2  # disgust
                text_vec[2] += item["score"] * 0.2  # fear
                text_vec[4] += item["score"] * 0.3  # sad
            elif item["label"].lower() == "positive":
                # Distribui entre happy e surprise
                text_vec[3] += item["score"] * 0.7  # happy
                text_vec[5] += item["score"] * 0.3  # surprise
            else:  # neutral
                text_vec[6] += item["score"]  # neutral
        
        # Normaliza o vetor
        if text_vec.sum() > 0:
            text_vec = text_vec / text_vec.sum()
        else:
            text_vec = np.ones(7, dtype=np.float32) / 7
            
        # Adiciona pequeno ruído para evitar zeros
        text_vec = text_vec + 1e-6
        text_vec = text_vec / text_vec.sum()

    except Exception as e:
        print(f"Erro na análise de texto: {e}")
        # Retorna distribuição uniforme em caso de erro
        text_vec = np.ones(7, dtype=np.float32) / 7

    return text_vec

# %% [markdown]
"""
## 5. Modelo de Fusão de IA

Rede neural em PyTorch que recebe um vetor concat (21-dim) e gera uma previsão final (7-dim).
O treinamento é contínuo, com pseudo‑label igual à média dos 3 vetores unimodais.
"""
# %%
class FusionModel(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=256, output_dim=7):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, x):
        # Adiciona uma dimensão extra para batch_size=1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        return F.softmax(self.fc3(x), dim=1)


fusion_model = FusionModel().to(DEVICE).float()
optimizer = optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.KLDivLoss(reduction='batchmean')


def update_fusion_model(face_vec, audio_vec, text_vec):
    """
    Atualiza o modelo de fusão com os vetores de entrada e retorna o vetor de saída (7-dim).
    """
    # Normaliza os vetores de entrada
    face_vec = face_vec / (np.sum(face_vec) if np.sum(face_vec) > 0 else 1)
    audio_vec = audio_vec / (np.sum(audio_vec) if np.sum(audio_vec) > 0 else 1)
    text_vec = text_vec / (np.sum(text_vec) if np.sum(text_vec) > 0 else 1)
    
    # Concatena os vetores e garante que seja um tensor 2D
    inp = np.concatenate([face_vec, audio_vec, text_vec]).astype(np.float32)
    inp_t = torch.tensor(inp, device=DEVICE).unsqueeze(0).float()  # Adiciona dimensão de batch
    
    # Calcula o target com pesos diferentes para cada modalidade
    weights = np.array([0.4, 0.3, 0.3])  # Pesos para face, áudio e texto
    target = weights[0] * face_vec + weights[1] * audio_vec + weights[2] * text_vec
    target = target / (target.sum() if target.sum() > 0 else 1)
    tgt_t = torch.tensor(target, device=DEVICE).unsqueeze(0).float()  # Adiciona dimensão de batch

    fusion_model.train()
    with torch.cuda.amp.autocast():
        out = fusion_model(inp_t)

    loss = loss_fn(torch.log(out + 1e-8), tgt_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del inp_t, tgt_t, loss
    torch.cuda.empty_cache()
    gc.collect()

    return out.detach().cpu().numpy().squeeze()

# %% [markdown]
"""
## 6. Processamento de Segmentos com Fusão

Para cada segmento:
- Seleciona frame no meio do segmento.
- Extrai snippet de áudio.
- Executa análises facial, áudio, texto.
- Atualiza modelo de fusão e padrões do locutor.
"""
# %%
def get_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame, fps

def detect_speech_pause(audio_data, threshold=0.1, min_duration=0.5):
    """
    Detecta pausas na fala baseado na energia do sinal de áudio.
    
    Args:
        audio_data: Dados de áudio em formato numpy array
        threshold: Limiar de energia para considerar como pausa
        min_duration: Duração mínima em segundos para considerar uma pausa
        
    Returns:
        Lista de tuplas (início, fim) das pausas detectadas
    """
    # Calcula a energia do sinal
    energy = np.abs(audio_data)
    # Normaliza
    energy = energy / np.max(energy)
    
    # Encontra segmentos com energia abaixo do limiar
    is_pause = energy < threshold
    # Converte para índices de início e fim
    changes = np.diff(is_pause.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # Ajusta os índices para considerar o início e fim do sinal
    if len(starts) > 0 and len(ends) > 0:
        if ends[0] < starts[0]:
            starts = np.insert(starts, 0, 0)
        if starts[-1] > ends[-1]:
            ends = np.append(ends, len(energy))
    
    # Converte para segundos e filtra pausas muito curtas
    sample_rate = 16000  # Taxa de amostragem do áudio
    pauses = []
    for start, end in zip(starts, ends):
        duration = (end - start) / sample_rate
        if duration >= min_duration:
            pauses.append((start/sample_rate, end/sample_rate))
    
    return pauses

def process_segment_with_speech(segment, video_path, audio_file, speaker_patterns):
    """
    Processa um segmento com fala, realizando análise completa.
    """
    start, end, speaker = segment["start"], segment["end"], segment["speaker"]
    mid_time = (start + end) / 2
    frame, _ = get_frame_at_time(video_path, mid_time)

    # Análise facial
    face_vec = analyze_face_emotion_vector(frame)

    # Extrai snippet de áudio
    snippet_path = f"snippet_{start:.2f}_{end:.2f}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file,
        "-ss", str(start), "-to", str(end),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        snippet_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Análise de áudio
    audio_vec = analyze_audio_emotion(snippet_path)

    # Transcrição e análise de texto
    transcript = transcribe_audio(snippet_path)
    text_vec = analyze_text_emotion(transcript)

    # Limpa arquivo temporário
    if os.path.exists(snippet_path):
        os.remove(snippet_path)

    # Fusão e atualização do modelo
    fused_vec = update_fusion_model(face_vec, audio_vec, text_vec)
    emo_idx = int(np.argmax(fused_vec))
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    fused_emotion = emotion_labels[emo_idx]

    speaker_patterns.setdefault(speaker, []).append((start, end, fused_emotion))

    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "face_vec": face_vec.tolist(),
        "audio_vec": audio_vec.tolist(),
        "text_vec": text_vec.tolist(),
        "transcript": transcript,
        "fused_vec": fused_vec.tolist(),
        "fused_emotion": fused_emotion,
        "has_speech": True
    }

def process_segment_without_speech(segment, video_path):
    """
    Processa um segmento sem fala, realizando apenas análise facial.
    """
    start, end = segment["start"], segment["end"]
    mid_time = (start + end) / 2
    frame, _ = get_frame_at_time(video_path, mid_time)

    # Apenas análise facial
    face_vec = analyze_face_emotion_vector(frame)
    emo_idx = int(np.argmax(face_vec))
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion = emotion_labels[emo_idx]

    return {
        "start": start,
        "end": end,
        "speaker": "no_speech",
        "face_vec": face_vec.tolist(),
        "audio_vec": [0.0] * 7,
        "text_vec": [0.0] * 7,
        "transcript": "",
        "fused_vec": face_vec.tolist(),
        "fused_emotion": emotion,
        "has_speech": False
    }

# %% [markdown]
"""
## 7. Pipeline Offline Completo

- Extrai áudio.
- Diariza.
- Processa segmentos.
- Agrega resultados por locutor e patterns.
"""
# %%
def process_speaker_data(results, speaker_patterns):
    """
    Processa os resultados e padrões dos falantes para gerar uma saída estruturada.
    
    Args:
        results: Lista de resultados dos segmentos processados
        speaker_patterns: Dicionário com padrões de emoção por falante
        
    Returns:
        Lista de dicionários com dados processados por falante
    """
    speakers_data = {}
    for res in results:
        spk = res["speaker"]
        speakers_data.setdefault(spk, {"segmentos": [], "emocao_segmentos": [], "padroes": [], "raw_analysis": []})
        speakers_data[spk]["segmentos"].append({"inicio": res["start"], "fim": res["end"]})
        speakers_data[spk]["emocao_segmentos"].append({
            "tempo": [res["start"], res["end"]],
            "emocao": res["fused_emotion"],
            "vetor": res["fused_vec"]
        })
        speakers_data[spk]["raw_analysis"].append(res)

    for spk, segs in speaker_patterns.items():
        emotions = [emo for (_, _, emo) in segs]
        for i in range(len(emotions) - 2):
            if emotions[i] == emotions[i + 1] == emotions[i + 2]:
                pattern = f"Emoção consistente '{emotions[i]}' nos segmentos {i+1}-{i+3}"
                speakers_data[spk]["padroes"].append(pattern)

    output = []
    for spk, data in speakers_data.items():
        counts = {}
        for seg in data["emocao_segmentos"]:
            counts[seg["emocao"]] = counts.get(seg["emocao"], 0) + 1
        dominant_emotion = max(counts, key=counts.get) if counts else "unknown"
        output.append({
            "pessoa": spk,
            "segmentos": data["segmentos"],
            "emocao_dominante": dominant_emotion,
            "emocao_segmentos": data["emocao_segmentos"],
            "padroes": data["padroes"],
            "raw_analysis": data["raw_analysis"]
        })

    return output

def process_video_offline_fusion(video_path, hf_token):
    try:
        print("Extraindo áudio do vídeo...")
        video_dir = os.path.dirname(video_path)
        audio_file = extract_audio(video_path, os.path.join(video_dir, "extracted_audio.wav"))

        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado após extração: {audio_file}")

        print("Executando diarização...")
        segments = perform_diarization(audio_file, hf_token)

        results = []
        speaker_patterns = {}
        print("Processando segmentos com fusão...")

        # Processa segmentos com fala
        for seg in tqdm(segments, desc="Processando segmentos com fala"):
            print(f"\nProcessando segmento com fala: {seg}")
            res = process_segment_with_speech(seg, video_path, audio_file, speaker_patterns)
            print(f"Resultado: {res}")
            results.append(res)

        # Processa segmentos sem fala (entre os segmentos de fala)
        speech_segments = sorted(segments, key=lambda x: x["start"])
        for i in range(len(speech_segments) - 1):
            current_end = speech_segments[i]["end"]
            next_start = speech_segments[i + 1]["start"]
            if next_start - current_end > 0.5:  # Pausa maior que 0.5 segundos
                pause_segment = {
                    "start": current_end,
                    "end": next_start,
                    "speaker": "no_speech"
                }
                print(f"\nProcessando segmento sem fala: {pause_segment}")
                res = process_segment_without_speech(pause_segment, video_path)
                results.append(res)

        if os.path.exists(audio_file):
            os.remove(audio_file)

        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()
        out.release()  # Fecha o writer de vídeo

        return process_speaker_data(results, speaker_patterns)

    except Exception as e:
        print(f"Erro no processamento do vídeo: {str(e)}")
        raise

# %% [markdown]
"""
## 8. Pipeline de Streaming Completo

Captura em tempo real via PyAudio e OpenCV, processa em blocos de `duration` segundos.
"""
# %%
def process_streaming_fusion(duration=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    # Definindo nomes dos arquivos temporários
    temp_video_file = "temp_stream.mp4"
    audio_filename = "temp_audio.wav"

    audio_buffer = queue.Queue()
    speech_buffer = []

    def audio_callback(in_data, frame_count, time_info, status):
        audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    cap = cv2.VideoCapture(0)
    captured_frames = []
    start_time = time.time()
    segment_results = []
    speaker_patterns = {}

    # Configuração do writer de vídeo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, fps, (frame_width, frame_height))

    print("Iniciando streaming multimodal. Pressione 'q' na janela de vídeo para encerrar.")

    while True:
        ret, frame = cap.read()
        if ret:
            captured_frames.append(frame)
            out.write(frame)  # Salva o frame no arquivo de vídeo
            cv2.imshow("Streaming", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time >= duration:
            frames_audio = []
            while not audio_buffer.empty():
                frames_audio.append(audio_buffer.get())
            
            if frames_audio:
                # Salva o áudio em um arquivo WAV
                wf = wave.open(audio_filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames_audio))
                wf.close()

                audio_data = np.frombuffer(b''.join(frames_audio), dtype=np.int16)
                pauses = detect_speech_pause(audio_data)
                
                if pauses:
                    # Processa segmentos entre pausas
                    last_end = 0
                    for pause_start, pause_end in pauses:
                        if pause_start - last_end > 0.5:  # Segmento com fala
                            segment = {
                                "start": last_end,
                                "end": pause_start,
                                "speaker": "stream_speaker"
                            }
                            res = process_segment_with_speech(segment, temp_video_file, audio_filename, speaker_patterns)
                            segment_results.append(res)
                        
                        # Processa segmento sem fala (pausa)
                        pause_segment = {
                            "start": pause_start,
                            "end": pause_end,
                            "speaker": "no_speech"
                        }
                        res = process_segment_without_speech(pause_segment, temp_video_file)
                        segment_results.append(res)
                        last_end = pause_end
                    
                    # Processa último segmento se houver
                    if last_end < duration:
                        segment = {
                            "start": last_end,
                            "end": duration,
                            "speaker": "stream_speaker"
                        }
                        res = process_segment_with_speech(segment, temp_video_file, audio_filename, speaker_patterns)
                        segment_results.append(res)
                else:
                    # Se não detectou pausas, processa como um único segmento com fala
                    segment = {
                        "start": 0,
                        "end": duration,
                        "speaker": "stream_speaker"
                    }
                    res = process_segment_with_speech(segment, temp_video_file, audio_filename, speaker_patterns)
                    segment_results.append(res)

            start_time = time.time()
            captured_frames = []
            if os.path.exists(audio_filename):
                os.remove(audio_filename)
            if os.path.exists(temp_video_file):
                os.remove(temp_video_file)

    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()
    out.release()  # Fecha o writer de vídeo

    return process_speaker_data(segment_results, speaker_patterns)

# %% [markdown]
"""
## 9. Execução Principal

Escolha o modo de execução:
- `"offline"`: Processa um vídeo gravado (defina o caminho do vídeo).
- `"streaming"`: Executa a captura ao vivo (pressione 'q' na janela de vídeo para encerrar).

A saída final é impressa em JSON.
"""
# %%
if __name__ == "__main__":
    mode = "offline"  # Altere para "streaming" para testar o modo ao vivo.

    if mode == "offline":
        video_file = "/home/joao/yo/multimodal-sentiment-analyzer/teste.mp4"  # Defina seu caminho
        resultado = process_video_offline_fusion(video_file, HF_TOKEN)
        print("Resultado JSON:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    else:
        resultado = process_streaming_fusion(duration=5)
        print("Resultado JSON:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))

# %% [markdown]
"""
# Documentação das Funções Principais

## Funções de Processamento de Vídeo

### `extract_audio(video_path: str, audio_output: str) -> str`
Extrai o áudio do vídeo para análise.

### `load_video_frames(video_path: str) -> (List[np.ndarray], List[float], float)`
Extrai frames e timestamps.

## Funções de Processamento de Áudio

### `analyze_audio_emotion(audio_segment_path: str) -> np.ndarray`
Retorna vetor de emoções (7-dim).

## Funções de Transcrição e Diarização

### `transcribe_audio(audio_segment_path: str) -> str`
Transcreve o áudio via Whisper.

### `perform_diarization(audio_path: str, hf_token: str) -> List[Dict]`
Segmenta áudio por locutor.

## Funções de Análise de Sentimento

### `analyze_face_emotion_vector(frame: np.ndarray) -> np.ndarray`
Analisa emoções faciais.

### `analyze_text_emotion(text: str) -> np.ndarray`
Analisa emoções do texto.

### `update_fusion_model(face_vec: np.ndarray, audio_vec: np.ndarray, text_vec: np.ndarray) -> np.ndarray`
Treina o modelo de fusão em loop contínuo.

## Pipelines

- `process_video_offline_fusion`
- `process_streaming_fusion`
"""
