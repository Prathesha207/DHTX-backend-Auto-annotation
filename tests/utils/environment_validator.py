import os
import sys
import socket
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EnvironmentValidator")

def check_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available according to PyTorch.")
            return False
        
        logger.info(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch version: {torch.__version__}")
        return True
    except ImportError:
        logger.error("PyTorch is not installed.")
        return False

def check_models():
    # Verify YOLO and Pose models exist
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "weights")
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found at {models_dir}")
        return True
    
    pt_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if not pt_files:
        logger.warning("No .pt model files found in models directory.")
        return True
    
    logger.info(f"Found {len(pt_files)} model files.")
    return True

def check_directories_writable():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    dirs_to_check = {
        "UPLOAD_DIR": os.path.join(base_dir, "uploads"),
        "OUTPUT_DIR": os.path.join(base_dir, "outputs"),
        "DB_DIR": base_dir
    }
    
    for name, d in dirs_to_check.items():
        if not d:
            continue
        os.makedirs(d, exist_ok=True)
        if not os.access(d, os.W_OK):
            logger.error(f"Directory {name} ({d}) is not writable.")
            return False
        logger.info(f"Directory {name} ({d}) is writable.")
        
    return True

def check_port(port=8000):
    # Check if port is in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('127.0.0.1', port))
        if result == 0:
            logger.error(f"Port {port} is currently in use! Please stop the existing backend server before running tests.")
            return False
        logger.info(f"Port {port} is available for the test backend.")
    return True

def run_all_checks():
    logger.info("Starting Environment Validation...")
    
    success = True
    success &= check_cuda()
    success &= check_models()
    success &= check_directories_writable()
    success &= check_port(8000)
    
    if not success:
        logger.error("Environment Validation FAILED. Please fix the issues above before running tests.")
        sys.exit(1)
        
    logger.info("Environment Validation PASSED.")

if __name__ == "__main__":
    run_all_checks()
