import logging
import os

# Códigos ANSI para colores
RESET = "\033[0m"
COLORS = {
    'DEBUG': "\033[32m",    # Verde
    'INFO': "\033[97m",     # Blanco brillante
    'WARNING': "\033[33m",  # Amarillo
    'ERROR': "\033[31m",    # Rojo
    'CRITICAL': "\033[31m", # Rojo
}

# Formatter con colores para consola
class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{log_color}{message}{RESET}"

def setup_logging():
    project_root = os.getcwd()
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "airbriefer.log")

    logger = logging.getLogger("airbriefer")
    logger.setLevel(logging.DEBUG)  # Mostrará todos los niveles
    logger.handlers.clear()
    logger.propagate = False  # Evita logs duplicados

    # Handler para archivo (sin colores)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_formatter)

    # Handler para consola (con colores)
    console_formatter = ColorFormatter("%(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(f"Logging to {log_path}")
    return logger

# Inicialización
logger = setup_logging()

