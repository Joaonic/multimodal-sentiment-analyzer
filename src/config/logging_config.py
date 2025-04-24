import logging
import os
from datetime import datetime

def setup_logging():
    """Configura o logging para o projeto"""
    # Cria o diretório de logs se não existir
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configura o formato do log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configura o arquivo de log
    log_file = os.path.join(log_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configura o console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configura o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configura loggers específicos
    analyzers_logger = logging.getLogger("src.analyzers")
    analyzers_logger.setLevel(logging.DEBUG)
    
    processors_logger = logging.getLogger("src.processors")
    processors_logger.setLevel(logging.DEBUG)
    
    models_logger = logging.getLogger("src.models")
    models_logger.setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado. Arquivo de log: {log_file}")
    
    return log_file 