import argparse
import socket
from pathlib import Path
from src.api.app import start_server
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def find_available_port(start_port: int = 8000, max_tries: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_tries} attempts")

def main():
    parser = argparse.ArgumentParser(description='Start the prediction API server')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path("model_checkpoints/best_model.pth")
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Please train the model first.")
        logger.info("Run 'python -m src.main' to train the model.")
        return
    
    # Find available port if none specified
    port = args.port if args.port is not None else find_available_port()
    
    logger.info(f"Starting server on {args.host}:{port}")
    try:
        start_server(host=args.host, port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main() 