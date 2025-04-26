import logging
import os
from datetime import datetime

def setup_logger(name: str = "prediccion_de_decisiones", log_dir: str = "./logs", level: int = logging.DEBUG) -> logging.Logger:
    """
    Configura un logger con salida a consola y a un archivo de log por fecha.

    Args:
        name (str): Nombre del logger.
        log_dir (str): Directorio donde se guardarán los logs.
        level (int): Nivel mínimo de logs para el logger.

    Returns:
        logging.Logger: Instancia del logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evita añadir múltiples handlers si ya están configurados
    if logger.hasHandlers():
        return logger

    # Crear directorio de logs si no existe
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir), exist_ok=True)

    # Generar nombre de archivo basado en la fecha
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{date_str}.log")

    # Formato de logs
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Asignar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
