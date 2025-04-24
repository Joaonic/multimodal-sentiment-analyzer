import logging
from typing import Optional
import os

import numpy as np
import torch
import torchaudio
from speechbrain.inference import foreign_class
from src.structures.analysis import AudioAnalysis
from src.utils.normalization import AudioFeatureNormalizer

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000
    ):
        """
        Inicializa o analisador de áudio.
        
        Args:
            device: Dispositivo para execução ('cuda' ou 'cpu')
            sample_rate: Taxa de amostragem do áudio
        """
        self.device = device
        self.sample_rate = sample_rate
        self.normalizer = AudioFeatureNormalizer(device)
        logger.info(f"Inicializando AudioAnalyzer no dispositivo: {device}, sample_rate: {sample_rate}")
        
        # Modelo de emoção com 8 classes
        self.emotion_model = foreign_class(
            source=os.getenv("AUDIO_MODEL", "speechbrain/emotion-recognition-wav2vec2-iemocap"),
            savedir="pretrained_models/ser",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device}
        )
        logger.info(f"Modelo de emoção configurado: {os.getenv('AUDIO_MODEL', 'speechbrain/emotion-recognition-wav2vec2-iemocap')}")
        
        # Modelo para análise de pitch
        pitch_model_class = os.getenv("PITCH_MODEL", "torchaudio.transforms.PitchShift")
        self.pitch_model = getattr(torchaudio.transforms, pitch_model_class.split(".")[-1])(
            sample_rate=sample_rate,
            n_steps=0
        )
        self.pitch_model.to(device)
        logger.info(f"Modelo de pitch configurado: {pitch_model_class}")
        
        # Configurações de análise
        self.window_size = 0.025  # 25ms
        self.hop_length = 0.010   # 10ms
        logger.info(f"Configurações de análise - window_size: {self.window_size}s, hop_length: {self.hop_length}s")
        
    def analyze(self, audio_path: str, speaker_id: str) -> AudioAnalysis:
        """
        Analisa o áudio e retorna todas as características.
        
        Args:
            audio_path: Caminho do arquivo de áudio
            speaker_id: Identificador do locutor
            
        Returns:
            Objeto AudioAnalysis com todas as características
        """
        try:
            logger.debug(f"Iniciando análise de áudio para speaker_id: {speaker_id}, arquivo: {audio_path}")
            
            # Carrega o áudio
            waveform, sr = torchaudio.load(audio_path)
            logger.debug(f"Áudio carregado - taxa de amostragem: {sr}, forma do waveform: {waveform.shape}")
            
            if sr != self.sample_rate:
                logger.debug(f"Reamostrando de {sr} para {self.sample_rate}")
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Move para o dispositivo
            waveform = waveform.to(self.device)
            
            # Análise de emoção
            logger.debug("Iniciando análise de emoção")
            emotion_probs = self._analyze_emotion(waveform)
            logger.debug(f"Dimensões do tensor de emoção: {emotion_probs.shape}")
            
            # Análise de pitch
            logger.debug("Iniciando análise de pitch")
            pitch = self._analyze_pitch(waveform)
            logger.debug(f"Dimensões do tensor de pitch: {pitch.shape}")
            
            # Análise de intensidade
            logger.debug("Iniciando análise de intensidade")
            intensity = self._analyze_intensity(waveform)
            logger.debug(f"Dimensões do tensor de intensidade: {intensity.shape}")
            
            # Análise de timbre
            logger.debug("Iniciando análise de timbre")
            timbre = self._analyze_timbre(waveform)
            logger.debug(f"Dimensões do tensor de timbre: {timbre.shape}")
            
            # Análise de velocidade da fala
            logger.debug("Iniciando análise de velocidade da fala")
            speech_rate = self._analyze_speech_rate(waveform)
            logger.debug(f"Dimensões do tensor de velocidade da fala: {speech_rate.shape}")
            
            # Análise de ritmo
            logger.debug("Iniciando análise de ritmo")
            rhythm = self._analyze_rhythm(waveform)
            logger.debug(f"Dimensões do tensor de ritmo: {rhythm.shape}")
            
            # Concatena todas as features
            features = torch.cat([
                emotion_probs,
                pitch,
                intensity,
                timbre,
                speech_rate,
                rhythm
            ], dim=1)
            
            # Normaliza as features
            features = self.normalizer.normalize(features)
            logger.debug(f"Dimensões após normalização: {features.shape}")
            
            # Calcula métricas de qualidade
            logger.debug("Calculando métricas de qualidade")
            audio_quality = self._calculate_audio_quality(waveform)
            signal_noise_ratio = self._calculate_signal_noise_ratio(waveform)
            clarity = self._calculate_clarity(waveform)
            consistency = self._calculate_consistency(waveform)
            
            logger.debug(f"Métricas de qualidade - Audio: {audio_quality:.2f}, SNR: {signal_noise_ratio:.2f}, Clareza: {clarity:.2f}, Consistência: {consistency:.2f}")
            
            return AudioAnalysis(
                speaker_id=speaker_id,
                emotion_probs=features[:, :8],  # Primeiras 8 dimensões são emoções
                pitch=features[:, 8:9],         # Próxima dimensão é pitch
                intensity=features[:, 9:10],    # Próxima dimensão é intensidade
                timbre=features[:, 10:23],      # Próximas 13 dimensões são timbre
                speech_rate=features[:, 23:24], # Próxima dimensão é velocidade da fala
                rhythm=features[:, 24:27],      # Próximas 3 dimensões são ritmo
                audio_quality=audio_quality,
                signal_noise_ratio=signal_noise_ratio,
                clarity=clarity,
                consistency=consistency
            )
        except Exception as e:
            logger.error(f"Erro na análise de áudio: {str(e)}", exc_info=True)
            return self._get_default_analysis(speaker_id)
    
    def _analyze_emotion(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa as emoções no áudio"""
        try:
            out_prob, _, _, _ = self.emotion_model.classify_batch(waveform)
            probs = out_prob.squeeze().float()
            
            # Garante que o tensor tem dimensão de batch
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
                
            # Se o modelo retornou 4 emoções, expande para 8
            if probs.shape[1] == 4:
                logger.warning("Modelo retornou 4 emoções, expandindo para 8")
                # Duplica as probabilidades para as 4 emoções adicionais
                probs = torch.cat([probs, probs], dim=1)
                # Normaliza para somar 1
                probs = probs / probs.sum(dim=1, keepdim=True)
                
            return probs
        except Exception as e:
            logger.error(f"Erro na análise de emoção: {e}")
            return torch.ones(1, 8, device=self.device).float() / 8  # Distribuição uniforme
    
    def _analyze_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa a entonação (pitch)"""
        try:
            # Usa o PitchShift para estimar o pitch
            pitch_shifted = self.pitch_model(waveform)
            # Calcula a diferença entre o original e o pitch shift
            pitch = torch.abs(waveform - pitch_shifted)
            # Normaliza o pitch
            pitch = (pitch - pitch.mean()) / (pitch.std() + 1e-6)
            # Retorna a média por canal e adiciona dimensão de batch
            return pitch.mean(dim=1).unsqueeze(0)  # [1, 1]
        except Exception as e:
            print(f"Erro na análise de pitch: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def _analyze_intensity(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa a intensidade do áudio"""
        try:
            # Calcula a energia do sinal
            energy = torch.sum(waveform ** 2, dim=1)
            # Normaliza
            energy = (energy - energy.mean()) / (energy.std() + 1e-6)
            # Adiciona dimensão de batch
            return energy.unsqueeze(0)  # [1, 1]
        except Exception as e:
            print(f"Erro na análise de intensidade: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def _analyze_timbre(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa o timbre do áudio"""
        try:
            # Calcula MFCCs (Mel-Frequency Cepstral Coefficients)
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13
            ).to(self.device)(waveform)
            # Normaliza
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            # Média por canal e adiciona dimensão de batch
            return mfcc.mean(dim=2).squeeze().unsqueeze(0)  # [1, 13]
        except Exception as e:
            print(f"Erro na análise de timbre: {e}")
            return torch.zeros(1, 13, device=self.device)
    
    def _analyze_speech_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa a velocidade da fala"""
        try:
            # Detecta silêncios
            energy = torch.sum(waveform ** 2, dim=1)
            threshold = energy.mean() * 0.1
            speech_segments = (energy > threshold).float()
            
            # Calcula a taxa de fala
            speech_rate = torch.sum(speech_segments) / len(speech_segments)
            # Adiciona dimensão de batch
            return speech_rate.unsqueeze(0).unsqueeze(0)  # [1, 1]
        except Exception as e:
            print(f"Erro na análise de velocidade: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def _analyze_rhythm(self, waveform: torch.Tensor) -> torch.Tensor:
        """Analisa o ritmo e pausas"""
        try:
            # Calcula a energia em janelas
            window_samples = int(self.window_size * self.sample_rate)
            hop_samples = int(self.hop_length * self.sample_rate)
            
            # Move o tensor para o dispositivo correto
            waveform = waveform.to(self.device)
            
            energy = torch.nn.functional.unfold(
                waveform.unsqueeze(0).unsqueeze(0),
                kernel_size=(1, window_samples),
                stride=(1, hop_samples)
            )
            energy = torch.sum(energy ** 2, dim=1)
            
            # Calcula estatísticas do ritmo
            rhythm_features = torch.cat([
                energy.mean(dim=1),
                energy.std(dim=1),
                torch.tensor([len(energy[0]) / self.sample_rate], device=self.device)  # Duração
            ])
            
            # Adiciona dimensão de batch
            return rhythm_features.unsqueeze(0)  # [1, 3]
        except Exception as e:
            print(f"Erro na análise de ritmo: {e}")
            return torch.zeros(1, 3, device=self.device)
    
    def _calculate_audio_quality(self, waveform: torch.Tensor) -> float:
        """Calcula a qualidade geral do áudio"""
        try:
            # Combina várias métricas para uma pontuação geral
            snr = self._calculate_signal_noise_ratio(waveform)
            clarity = self._calculate_clarity(waveform)
            consistency = self._calculate_consistency(waveform)
            
            # Ponderando as métricas
            return 0.4 * snr + 0.3 * clarity + 0.3 * consistency
        except:
            return 0.0
    
    def _calculate_signal_noise_ratio(self, waveform: torch.Tensor) -> float:
        """Calcula a relação sinal-ruído"""
        try:
            # Estima o ruído usando o primeiro e último 5% do sinal
            noise_samples = int(0.05 * waveform.shape[1])
            noise = torch.cat([waveform[:, :noise_samples], waveform[:, -noise_samples:]])
            noise_power = torch.mean(noise ** 2)
            
            # Potência do sinal
            signal_power = torch.mean(waveform ** 2)
            
            # SNR em dB
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-6))
            return min(max(snr.item() / 30, 0), 1)  # Normaliza para [0,1]
        except:
            return 0.0
    
    def _calculate_clarity(self, waveform: torch.Tensor) -> float:
        """Calcula a clareza do áudio"""
        try:
            # Calcula a energia em diferentes bandas de frequência
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13
            ).to(self.device)(waveform)
            
            # Calcula a razão entre as bandas de alta e baixa frequência
            high_freq = torch.mean(torch.abs(mfcc[:, 6:]))
            low_freq = torch.mean(torch.abs(mfcc[:, :6]))
            clarity = high_freq / (low_freq + 1e-6)
            
            return min(max(clarity.item(), 0), 1)
        except:
            return 0.0
    
    def _calculate_consistency(self, waveform: torch.Tensor) -> float:
        """Calcula a consistência do áudio"""
        try:
            # Divide o sinal em segmentos
            segment_size = int(0.1 * self.sample_rate)  # 100ms
            segments = waveform.unfold(1, segment_size, segment_size)
            
            # Calcula a energia de cada segmento
            energy = torch.mean(segments ** 2, dim=2)
            
            # Calcula o coeficiente de variação da energia
            cv = torch.std(energy) / (torch.mean(energy) + 1e-6)
            
            # Quanto menor o CV, mais consistente é o sinal
            return 1.0 - min(cv.item(), 1.0)
        except:
            return 0.0
    
    def _get_default_analysis(self, speaker_id: str) -> AudioAnalysis:
        """Retorna uma análise padrão para quando não detecta áudio"""
        return AudioAnalysis(
            speaker_id=speaker_id,
            emotion_probs=torch.ones(1, 8, device=self.device) / 8,
            pitch=torch.zeros(1, 1, device=self.device),
            intensity=torch.zeros(1, 1, device=self.device),
            timbre=torch.zeros(1, 13, device=self.device),
            speech_rate=torch.zeros(1, 1, device=self.device),
            rhythm=torch.zeros(1, 3, device=self.device),
            audio_quality=0.0,
            signal_noise_ratio=0.0,
            clarity=0.0,
            consistency=0.0
        ) 