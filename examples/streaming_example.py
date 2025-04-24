import os
from src.config import MODEL_CONFIG, STREAMING_CONFIG
from src.processors.streaming_processor import StreamingProcessor

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
    def process_result(result):
        # Aqui você pode adicionar lógica adicional
        # como salvar resultados, enviar para API, etc.
        print(f"Emoção detectada: {result['fused_emotion']}")
        print(f"Confiança: {result['fused_vec'][result['fused_emotion']]:.2f}")
    
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