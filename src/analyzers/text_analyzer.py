import logging
from typing import Tuple
import os

import torch
from src.structures.analysis import TextAnalysis
from src.utils.normalization import TextFeatureNormalizer
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa o analisador de texto.
        
        Args:
            device: Dispositivo para execução ('cuda' ou 'cpu')
        """
        self.device = device
        self.normalizer = TextFeatureNormalizer(device)
        logger.info(f"Inicializando TextAnalyzer no dispositivo: {device}")
        
        # Modelo de emoção
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=7
        ).to(device)
        logger.info("Modelo de emoção configurado: neuralmind/bert-base-portuguese-cased")
        
        # Tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        
        # Modelo para embedding contextual
        self.context_model = AutoModel.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        ).to(device)
        
        # Modelo de sarcasmo
        self.sarcasm_model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=2
        ).to(device)
        
        # Modelo de humor
        self.humor_model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=2
        ).to(device)
        
        # Modelo de sentimento
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=3
        ).to(device)
        
        logger.info("Todos os modelos BERT carregados e configurados")
    
    def analyze(self, text: str, speaker_id: str) -> TextAnalysis:
        """Analisa o texto e retorna todas as características"""
        if not text or len(text.strip()) == 0:
            logger.warning(f"Texto vazio para speaker_id: {speaker_id}")
            return self._get_default_analysis(speaker_id)
            
        try:
            logger.debug(f"Iniciando análise de texto para speaker_id: {speaker_id}")
            logger.debug(f"Texto a ser analisado: {text[:100]}...")
            
            # Análise de emoção
            logger.debug("Iniciando análise de emoção")
            emotion_probs = self._analyze_emotion(text)
            logger.debug(f"Dimensões do tensor de emoção: {emotion_probs.shape}")
            
            # Análise de sarcasmo
            logger.debug("Iniciando análise de sarcasmo")
            sarcasm_score = self._analyze_sarcasm(text)
            logger.debug(f"Dimensões do tensor de sarcasmo: {sarcasm_score.shape}")
            
            # Análise de humor
            logger.debug("Iniciando análise de humor")
            humor_score = self._analyze_humor(text)
            logger.debug(f"Dimensões do tensor de humor: {humor_score.shape}")
            
            # Análise de polaridade e intensidade
            logger.debug("Iniciando análise de sentimento")
            polarity, intensity = self._analyze_sentiment(text)
            logger.debug(f"Dimensões do tensor de sentimento: {polarity.shape}, {intensity.shape}")
            
            # Embedding contextual
            logger.debug("Iniciando extração de embedding contextual")
            context_embedding = self._get_context_embedding(text)
            logger.debug(f"Dimensões do tensor de embedding: {context_embedding.shape}")
            
            # Concatena todas as features
            features = torch.cat([
                emotion_probs,
                sarcasm_score,
                humor_score,
                polarity,
                intensity,
                context_embedding
            ], dim=1)
            
            # Normaliza as features
            features = self.normalizer.normalize(features)
            logger.debug(f"Dimensões após normalização: {features.shape}")
            
            # Calcula métricas de qualidade
            logger.debug("Calculando métricas de qualidade")
            text_quality = self._calculate_text_quality(text)
            coherence = self._calculate_coherence(text)
            completeness = self._calculate_completeness(text)
            relevance = self._calculate_relevance(text)
            
            logger.debug(f"Métricas de qualidade - Texto: {text_quality:.2f}, Coerência: {coherence:.2f}, Completude: {completeness:.2f}, Relevância: {relevance:.2f}")
            
            return TextAnalysis(
                speaker_id=speaker_id,
                emotion_probs=features[:, :7],         # Primeiras 7 dimensões são emoções
                sarcasm_score=features[:, 7:8],        # Próxima dimensão é sarcasmo
                humor_score=features[:, 8:9],          # Próxima dimensão é humor
                polarity=features[:, 9:10],            # Próxima dimensão é polaridade
                intensity=features[:, 10:11],          # Próxima dimensão é intensidade
                context_embedding=features[:, 11:779], # Próximas 768 dimensões são embedding
                text_quality=text_quality,
                coherence=coherence,
                completeness=completeness,
                relevance=relevance
            )
        except Exception as e:
            logger.error(f"Erro na análise de texto: {str(e)}", exc_info=True)
            return self._get_default_analysis(speaker_id)
    
    def _analyze_emotion(self, text: str) -> torch.Tensor:
        """Analisa as emoções no texto"""
        try:
            # Tokeniza o texto
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Obtém as probabilidades de emoção
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Garante 2D [1,7]
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            return probs  # shape [1,7]
        
        except Exception as e:
            print(f"Erro na análise de emoção: {e}")
            # Fallback: vetor uniforme de 7 emoções
            return torch.ones(1, 7, device=self.device) / 7
    
    def _analyze_sarcasm(self, text: str) -> torch.Tensor:
        """Analisa a probabilidade de sarcasmo"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.sarcasm_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                sarcasm_prob = probs[0][1]  # Probabilidade de sarcasmo
            return sarcasm_prob.unsqueeze(0).unsqueeze(0)  # [1, 1]
        except Exception as e:
            print(f"Erro na análise de sarcasmo: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def _analyze_humor(self, text: str) -> torch.Tensor:
        """Analisa a probabilidade de humor"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.humor_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                humor_prob = probs[0][1]  # Probabilidade de humor
            return humor_prob.unsqueeze(0).unsqueeze(0)  # [1, 1]
        except Exception as e:
            print(f"Erro na análise de humor: {e}")
            return torch.zeros(1, 1, device=self.device)
    
    def _analyze_sentiment(self, text: str) -> torch.Tensor:
        """Analisa o sentimento do texto"""
        try:
            # Tokeniza o texto
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Obtém o sentimento
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                sentiment = torch.softmax(outputs.logits, dim=-1)
            
            # Adiciona dimensão de batch
            return sentiment.unsqueeze(0)  # [1, 3]
        except Exception as e:
            print(f"Erro na análise de sentimento: {e}")
            return torch.ones(1, 3, device=self.device) / 3
    
    def _get_context_embedding(self, text: str) -> torch.Tensor:
        """Extrai o embedding contextual do texto"""
        try:
            # Tokeniza o texto
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Obtém os embeddings
            with torch.no_grad():
                outputs = self.context_model(**inputs)
                # Usa o embedding do token [CLS]
                embedding = outputs.last_hidden_state[:, 0, :]
            
            return embedding  # [1, 768]
        except Exception as e:
            print(f"Erro na extração de embedding: {e}")
            return torch.zeros(1, 768, device=self.device)  # Dimensão do BERT
    
    def _calculate_text_quality(self, text: str) -> float:
        """Calcula a qualidade geral do texto"""
        try:
            # Combina várias métricas para uma pontuação geral
            coherence = self._calculate_coherence(text)
            completeness = self._calculate_completeness(text)
            relevance = self._calculate_relevance(text)
            
            # Ponderando as métricas
            return 0.4 * coherence + 0.3 * completeness + 0.3 * relevance
        except:
            return 0.0
    
    def _calculate_coherence(self, text: str) -> float:
        """Calcula a coerência do texto"""
        try:
            # Tokeniza o texto
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Obtém os embeddings
            with torch.no_grad():
                outputs = self.context_model(**tokens)
                embeddings = outputs.last_hidden_state
            
            # Calcula a similaridade entre sentenças consecutivas
            similarities = []
            for i in range(embeddings.shape[1] - 1):
                sim = torch.cosine_similarity(
                    embeddings[:, i],
                    embeddings[:, i + 1],
                    dim=0
                )
                similarities.append(sim)
            
            # Média das similaridades
            return torch.mean(torch.tensor(similarities)).item()
        except:
            return 0.0
    
    def _calculate_completeness(self, text: str) -> float:
        """Calcula a completude do texto"""
        try:
            # Verifica se o texto contém elementos essenciais
            has_subject = len([t for t in text.split() if t.isalpha()]) > 0
            has_verb = len([t for t in text.split() if t.endswith('ar') or t.endswith('er') or t.endswith('ir')]) > 0
            has_punctuation = any(c in text for c in ['.', '!', '?'])
            
            # Combina os elementos
            completeness = 0.4 * has_subject + 0.4 * has_verb + 0.2 * has_punctuation
            return float(completeness)
        except:
            return 0.0
    
    def _calculate_relevance(self, text: str) -> float:
        """Calcula a relevância do texto"""
        try:
            # Verifica se o texto contém palavras-chave relevantes
            relevant_words = ['emoção', 'sentimento', 'expressão', 'reação', 'comportamento']
            word_count = sum(1 for word in relevant_words if word in text.lower())
            
            # Normaliza pelo número de palavras no texto
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
                
            return min(word_count / total_words, 1.0)
        except:
            return 0.0
    
    def _get_default_analysis(self, speaker_id: str) -> TextAnalysis:
        """Retorna uma análise padrão para quando não detecta texto"""
        return TextAnalysis(
            speaker_id=speaker_id,
            emotion_probs=torch.ones(1, 7, device=self.device) / 7,  # 7 emoções
            sarcasm_score=torch.zeros(1, 1, device=self.device),
            humor_score=torch.zeros(1, 1, device=self.device),
            polarity=torch.zeros(1, 1, device=self.device),
            intensity=torch.zeros(1, 1, device=self.device),
            context_embedding=torch.zeros(1, 768, device=self.device),
            text_quality=0.0,
            coherence=0.0,
            completeness=0.0,
            relevance=0.0
        ) 