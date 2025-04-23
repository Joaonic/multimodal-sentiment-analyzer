# Análise Multimodal de Sentimentos

Sistema para análise de sentimentos utilizando múltiplas modalidades (face, áudio e texto) baseado no dataset AMI.

## Estrutura do Projeto

```
.
├── data/
│   ├── ami_raw/          # Dados brutos do AMI
│   └── ami/              # Dados processados
├── checkpoints/          # Modelos treinados
├── fusion_model.py       # Modelo de fusão
├── train_fusion_model.py # Script de treinamento
├── inference.py          # Script de inferência
└── preprocess_ami.py     # Script de pré-processamento
```

## Requisitos

```bash
pip install torch torchvision torchaudio
pip install numpy pandas tqdm
pip install deepface
pip install speechbrain
pip install transformers
pip install opencv-python
pip install pyaudio
```

## Uso

### 1. Pré-processamento dos Dados

```bash
python preprocess_ami.py
```

Este script:
- Extrai vetores de emoção do dataset AMI
- Divide os dados em train/val/test
- Salva os dados processados em JSON

### 2. Treinamento do Modelo

```bash
python train_fusion_model.py
```

Este script:
- Carrega os dados processados
- Treina o modelo de fusão
- Salva checkpoints periodicamente
- Implementa early stopping

### 3. Inferência

```python
from inference import MultimodalAnalyzer

# Inicializa o analisador
analyzer = MultimodalAnalyzer('checkpoints/best_model.pt')

# Análise de segmento
face_vec = np.array([...])  # Vetor de emoções faciais
audio_vec = np.array([...]) # Vetor de emoções do áudio
text_vec = np.array([...])  # Vetor de emoções do texto

result = analyzer.analyze_segment(face_vec, audio_vec, text_vec)
print(result)

# Análise de vídeo completo
results = analyzer.analyze_video('video.mp4', 'results.json')

# Análise em streaming
def callback(results):
    print(results)

analyzer.analyze_streaming(duration=5.0, callback=callback)
```

## Modelo de Fusão

O modelo de fusão é uma rede neural que:
- Recebe vetores de emoção de 3 modalidades (21 dimensões)
- Combina as informações usando pesos aprendidos
- Produz uma distribuição de probabilidade sobre 7 emoções

### Arquitetura

- Camada de entrada: 21 dimensões
- Camada oculta 1: 256 unidades + ReLU + LayerNorm + Dropout
- Camada oculta 2: 128 unidades + ReLU + LayerNorm + Dropout
- Camada de saída: 7 unidades + Softmax

### Treinamento

- Loss: KL-Divergence
- Otimizador: AdamW com weight decay
- Early stopping baseado em validação
- Pesos iniciais das modalidades: face (0.4), áudio (0.3), texto (0.3)

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes. 