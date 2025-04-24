# Análise de Sentimentos Multimodal

Sistema de análise de sentimentos que combina informações de expressões faciais, áudio e texto para identificar emoções em vídeos e streaming em tempo real.

## 🚀 Visão Geral

O sistema utiliza três modalidades principais para análise de sentimentos:
- **Análise Facial**: Detecta expressões faciais, micro-expressões, direção do olhar e tensão muscular
- **Análise de Áudio**: Analisa emoções, pitch, intensidade, timbre, velocidade da fala e ritmo
- **Análise de Texto**: Processa emoções, sarcasmo, humor, polaridade e embedding contextual

Os resultados são combinados através de um modelo de fusão neural avançado que aprende continuamente com os dados.

## 📁 Estrutura do Projeto

```
multimodal-sentiment-analyzer/
├── data/                  # Dados brutos e processados
├── checkpoints/           # Modelos treinados
├── output/               # Resultados das análises
├── temp/                 # Arquivos temporários
├── src/
│   ├── analyzers/
│   │   ├── face_analyzer.py    # Análise de emoções faciais
│   │   ├── text_analyzer.py    # Análise de emoções textuais
│   │   └── audio_analyzer.py   # Análise de emoções no áudio
│   ├── models/
│   │   └── fusion_model.py     # Modelo de fusão multimodal
│   ├── processors/
│   │   ├── offline_processor.py    # Processamento de vídeos
│   │   └── streaming_processor.py  # Processamento em tempo real
│   ├── structures/
│   │   ├── analysis.py         # Estruturas de análise
│   │   ├── emotions.py         # Classes de emoções
│   │   ├── models.py           # Estruturas de modelos
│   │   └── config.py           # Estruturas de configuração
│   ├── config/
│   │   └── config.py           # Configurações do sistema
│   ├── training/               # Scripts de treinamento
│   ├── inference.py           # Script de inferência
│   └── main.py                # Script principal
├── .env                      # Variáveis de ambiente
├── requirements.txt          # Dependências Python
└── README.md                 # Documentação
```

## 🛠️ Requisitos

- Python 3.8+
- CUDA 11.8+ (para GPU)
- FFmpeg
- PortAudio

### Dependências Python

```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/multimodal-sentiment-analyzer.git
cd multimodal-sentiment-analyzer
```

2. Configure o token do HuggingFace no arquivo `.env`:
```bash
HF_TOKEN=seu_token_aqui
```

3. Ajuste as configurações em `src/config/config.py`:
- Diretórios de entrada/saída
- Modelos a serem utilizados
- Parâmetros de processamento
- Configurações de streaming

## 🎯 Uso

### Processamento de Vídeo

```python
from src.processors.offline_processor import OfflineProcessor
from src.config.config import MODEL_CONFIG, PROCESSING_CONFIG
import os

processor = OfflineProcessor(
    model_config=MODEL_CONFIG,
    processing_config=PROCESSING_CONFIG,
    hf_token=os.getenv('HF_TOKEN')
)

results = processor.process_video("video.mp4")
```

### Streaming em Tempo Real

```python
from src.processors.streaming_processor import StreamingProcessor
from src.config.config import MODEL_CONFIG, STREAMING_CONFIG

processor = StreamingProcessor(
    model_config=MODEL_CONFIG,
    streaming_config=STREAMING_CONFIG
)

def callback(result):
    print(f"Emoção detectada: {result['fused_emotion']}")

processor.run(duration=5.0, callback=callback)
```

## 📊 Saída

O sistema gera resultados estruturados incluindo:
- Análise facial: emoções, micro-expressões, direção do olhar, tensão muscular
- Análise de áudio: emoções, pitch, intensidade, timbre, velocidade, ritmo
- Análise de texto: emoções, sarcasmo, humor, polaridade, embedding
- Emoção final após fusão com confiança
- Timestamps e segmentos
- Padrões de emoção por falante
- Transcrições do áudio

Exemplo de saída:
```json
{
  "speaker_id": "SPEAKER_01",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 5.0,
      "face_analysis": {
        "emotion_probs": [0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0],
        "micro_expressions": [...],
        "gaze_direction": [...],
        "muscle_tension": [...],
        "movement_patterns": [...]
      },
      "audio_analysis": {
        "emotion_probs": [0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0],
        "pitch": 0.5,
        "intensity": 0.8,
        "timbre": [...],
        "speech_rate": 0.6,
        "rhythm": [...]
      },
      "text_analysis": {
        "emotion_probs": [0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0],
        "sarcasm_score": 0.1,
        "humor_score": 0.3,
        "polarity": 0.8,
        "intensity": 0.7,
        "context_embedding": [...]
      },
      "fused_analysis": {
        "emotion_probs": [0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0],
        "confidence": 0.85,
        "face_weight": 0.4,
        "audio_weight": 0.3,
        "text_weight": 0.3
      },
      "transcript": "Olá, como você está?",
      "confidence": 0.85,
      "dominant_emotion": "happy"
    }
  ],
  "dominant_emotion": "happy",
  "emotion_patterns": [
    "Emoção consistente 'happy' nos segmentos 1-3"
  ],
  "average_confidence": 0.85,
  "emotion_timeline": [
    {
      "time": 0.0,
      "emotion": "happy",
      "confidence": 0.85
    }
  ]
}
```

## 🤖 Modelo de Fusão

O modelo de fusão é uma rede neural avançada com:
- Processamento individual para cada modalidade
- Camadas de normalização e dropout
- Fusão adaptativa com pesos aprendidos
- Suporte para características adicionais
- Treinamento contínuo com pseudo-labels

## 🔄 Treinamento Contínuo

O modelo aprende continuamente através de:
1. Pseudo-labels gerados pela média ponderada das modalidades
2. Pesos adaptativos para cada modalidade
3. Validação com early stopping
4. Checkpointing automático
5. Monitoramento de métricas

## 📝 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request 