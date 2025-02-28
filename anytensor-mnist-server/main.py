from anytensor.core import TensorContext, GraphProgram, execute_graph_on_gpu
from tinygrad import Tensor, dtypes
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import safe_load
from tinygrad import nn
from typing import List, Callable
from rich import print
from rich.logging import RichHandler
import logging

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("mnist-server")

# For Processing Images
import numpy as np
import base64


# This is where the model is defined!!
class Model:
    def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.relu,
            nn.Conv2d(32, 32, 5),
            Tensor.relu,
            nn.BatchNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            nn.Conv2d(64, 64, 3),
            Tensor.relu,
            nn.BatchNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

    def load(self, weights):
        self.__dict__.update(weights)


def preprocess_image_numpy(image_bytes):
    """Process image bytes to a normalized numpy array of shape (1, 1, 28, 28)"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Check if this is a PNG/JPG (attempt to decode)
    try:
        # If image is in a common format (PNG/JPG), try to decode it
        import cv2  # Only import if needed

        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
    except:
        log.error("Failed to decode image")
        # If decoding fails, assume it's already a raw array format
        img = nparr.reshape(28, 28)

    # Normalize to 0-1 range
    img_norm = img.astype(np.float32) / 255.0

    # Reshape to (1, 1, 28, 28) - adding batch and channel dimensions
    return img_norm.reshape(1, 1, 28, 28)


# For inference, set the model to evaluation mode
@Tensor.test()
def compile_model(model: Callable[[Tensor], Tensor]) -> GraphProgram:
    # Make sure input data has the right shape (N, 1, 28, 28)
    # Assuming input_data is already a Tensor with the right shape
    log.info("Compiling model to graph program")
    context = TensorContext()
    input_image = context.add_graph_input("input_image", (1, 1, 28, 28), dtypes.float32)
    predictions = model(input_image)
    graph_program = context.compile_to_graph(predictions)
    if isinstance(graph_program, ValueError):
        log.error(f"Error compiling model: {graph_program}")
        raise ValueError(f"Error compiling model: {graph_program}")
    log.info("Model compiled successfully")
    return graph_program


def regular_inference(model: Callable[[Tensor], Tensor], input_data: Tensor) -> Tensor:
    # Run the model and get predictions
    predictions = model(input_data)
    # Get the predicted class (index of highest probability)
    return predictions.argmax(axis=1)


def infer_on_image_regular(image_path):
    log.info("Running regular TinyGrad inference with fresh model")

    # Load and process the image using NumPy
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Log image hash for debugging
    import hashlib

    img_hash = hashlib.md5(img_bytes).hexdigest()
    log.info(f"Processing image with hash: {img_hash}")

    # Convert image bytes to numpy array
    img_array = preprocess_image_numpy(img_bytes)
    log.info(
        f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}"
    )

    # Create tensor from preprocessed array
    img_tensor = Tensor(img_array)

    # Create a fresh model instance
    log.info("Creating fresh model instance")
    fresh_model = Model()

    # Load the weights into the fresh model
    log.info("Loading weights into fresh model")
    weights = safe_load("./mnist.safetensors")
    fresh_model.load(weights)

    # Run inference with the fresh model
    with Tensor.test():
        predictions = fresh_model(img_tensor)
        predicted_class = predictions.argmax(axis=1).item()

    log.info(f"Predicted digit: {predicted_class}")
    return predicted_class


def infer_on_image_anytensor(image_path):
    log.info("Running Anytensor graph inference with fresh model")

    # Load and process the image using NumPy
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Log image hash for debugging
    import hashlib

    img_hash = hashlib.md5(img_bytes).hexdigest()
    log.info(f"Processing image with hash: {img_hash}")

    # Convert image bytes to numpy array
    img_array = preprocess_image_numpy(img_bytes)
    log.info(
        f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}"
    )

    # Create tensor from preprocessed array
    img_tensor = Tensor(img_array)

    # Create a fresh model instance
    log.info("Creating fresh model instance for AnyTensor")
    fresh_model = Model()

    # Load the weights into the fresh model
    log.info("Loading weights into fresh model")
    weights = safe_load("./mnist.safetensors")
    fresh_model.load(weights)

    # Create a fresh graph program
    log.info("Compiling fresh graph program")
    context = TensorContext()
    input_image = context.add_graph_input("input_image", (1, 1, 28, 28), dtypes.float32)
    # Load the pre-compiled graph program from file
    with open("./mnist_program.eigentensor", "rb") as f:
        graph_program = GraphProgram.from_bytes(f.read())
    if isinstance(graph_program, ValueError):
        log.error(f"Error loading graph program: {graph_program}")
        raise ValueError(f"Error loading graph program: {graph_program}")

    # Execute graph with the current image data
    log.info("Executing fresh graph program")
    result = execute_graph_on_gpu(graph_program, {"input_image": img_tensor})
    predictions = result.argmax(axis=1)

    log.info(f"Predicted digit: {predictions.item()}")
    return predictions.item()


###
# MNIST Prediction Server
###

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Union
from io import BytesIO
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Compile model and store in dictionary before app starts
    log.info("Starting server initialization")
    mnist_program = compile_model(Model())

    # Store the compiled graph program to a file
    log.info("Writing graph program to ./mnist_program.eigentensor")
    with open("./mnist_program.eigentensor", "wb") as f:
        f.write(mnist_program.to_bytes())

    log.info("Server initialization complete")
    yield


app = FastAPI(lifespan=lifespan)


class SuccessResponse(BaseModel):
    status: Literal["success"]
    data: int


class ErrorResponse(BaseModel):
    status: Literal["error"]
    message: str


class InferenceRequest(BaseModel):
    image_bytes: str
    mode: Literal["regular", "anytensor"]


@app.post("/infer", response_model=Union[SuccessResponse, ErrorResponse])
async def infer(request: InferenceRequest):
    try:
        log.info(f"Received inference request using {request.mode} mode")
        # Decode base64 string to bytes
        image_data = base64.b64decode(request.image_bytes)

        # Save received image to disk
        with open("./current_image.png", "wb") as f:
            f.write(image_data)
        log.info("Saved received image to ./current_image.png")

        # Run appropriate inference using the saved image file
        if request.mode == "regular":
            prediction = infer_on_image_regular("./current_image.png")
        else:
            prediction = infer_on_image_anytensor("./current_image.png")

        return SuccessResponse(status="success", data=prediction)

    except Exception as e:
        log.error(f"Error during inference: {str(e)}")
        raise HTTPException(
            status_code=400, detail=ErrorResponse(status="error", message=str(e)).dict()
        )


if __name__ == "__main__":
    import uvicorn

    log.info("Starting MNIST server on port 8989")
    uvicorn.run(app, host="0.0.0.0", port=8989)