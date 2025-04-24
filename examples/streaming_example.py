import os
from src.config import MODEL_CONFIG, STREAMING_CONFIG
from src.processors.streaming_processor import StreamingProcessor
from typing import Dict
import numpy as np

def main():
    """
    Exemplo de uso do processador de streaming com visualização.
    """
    # Inicializa processador
    processor = StreamingProcessor(
        model_config=MODEL_CONFIG,
        streaming_config=STREAMING_CONFIG
    )
    
    # Função de callback para processamento adicional
    def process_result(result: Dict):
        """Processa os resultados da análise"""
        try:
            if result["fused_emotion"] is not None:
                # Obtém a emoção dominante
                emotion_idx = np.argmax(result["fused_emotion"])
                emotions = ["feliz", "triste", "raiva", "medo", "surpresa", "nojo", "neutro"]
                emotion = emotions[emotion_idx]
                confidence = float(result["fused_emotion"][emotion_idx])
                
                print(f"Emoção detectada: {emotion}")
                print(f"Confiança: {confidence:.2f}")
                
                # Mostra os pesos de cada modalidade
                if result["weights"] is not None:
                    print("\nPesos das modalidades:")
                    print(f"  Face:  {result['weights']['face']:.2f}")
                    print(f"  Áudio: {result['weights']['audio']:.2f}")
                    print(f"  Texto: {result['weights']['text']:.2f}")
                    
                # Mostra o speaker_id
                if result["speaker_id"] is not None:
                    print(f"\nSpeaker: {result['speaker_id']}")
                
            else:
                print("Nenhuma emoção detectada")
            
        except Exception as e:
            print(f"Erro ao processar resultado: {str(e)}")
            import traceback
            traceback.print_exc()
    
    try:
        # Inicia processamento
        print("Iniciando processamento de streaming...")
        print("Pressione 'q' para sair")
        processor.run(duration=5.0, callback=process_result)
        
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário")
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
    finally:
        print("Processamento finalizado")

if __name__ == "__main__":
    main() 