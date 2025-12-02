import logging
import sys

def configure_logger():
    """Configures and returns a logger for the finance application."""
    
    logger = logging.getLogger("finance_logger")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s') # Formato do log
        
        file_handler = logging.FileHandler('logs/finance_app.log') # Onde os logs serão salvos
        file_handler.setFormatter(formatter) 
        
        stream_handler = logging.StreamHandler(sys.stdout) # Log no console
        stream_handler.setFormatter(formatter)
        
        # Adiciona os handlers ao logger
        logger.addHandler(file_handler) 
        logger.addHandler(stream_handler)
        
    return logger

logger = configure_logger()