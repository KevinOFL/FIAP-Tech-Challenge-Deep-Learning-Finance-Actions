import logging
import sys
import os

def configure_logger(aplication_name: str) -> logging.Logger:
    """Configures and returns a logger for the finance application."""
    
    logger = logging.getLogger(aplication_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(message)s') # Formato do log
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Pega a raiz do projeto
        log_path = os.path.join(base_dir, 'logs', f'{aplication_name}.log')
        
        # Garante que a pasta logs existe
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_path) # Onde os logs serão salvos
        file_handler.setFormatter(formatter) 
        
        stream_handler = logging.StreamHandler(sys.stdout) # Log no console
        stream_handler.setFormatter(formatter)
        
        # Adiciona os handlers ao logger
        logger.addHandler(file_handler) 
        logger.addHandler(stream_handler)
        
    return logger

logger_api = configure_logger("api")
logger_db = configure_logger("database")
logger_training = configure_logger("training")