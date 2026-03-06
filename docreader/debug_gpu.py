
import logging
import sys
import os

# Configure logging to stdout
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_gpu")

print("Starting debug script...", flush=True)

try:
    import paddle
    print(f"Paddle version: {paddle.__version__}", flush=True)
    
    if paddle.is_compiled_with_cuda():
        print("CUDA is available", flush=True)
        try:
            paddle.device.set_device("gpu")
            print("Successfully set device to GPU", flush=True)
        except Exception as e:
            print(f"Failed to set GPU: {e}", flush=True)
    else:
        print("CUDA is NOT available", flush=True)

    print("Importing PaddleOCR...", flush=True)
    from paddleocr import PaddleOCR
    
    print("Initializing PaddleOCR...", flush=True)
    ocr = PaddleOCR(
        use_gpu=True, 
        lang="ch", 
        use_angle_cls=True,
        show_log=True
    )
    print("PaddleOCR initialized successfully", flush=True)

except Exception as e:
    print(f"An error occurred: {e}", flush=True)
    import traceback
    traceback.print_exc()
