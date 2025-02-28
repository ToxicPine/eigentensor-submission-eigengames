from anytensor.core import TensorContext, GraphProgram, execute_graph_on_gpu
from tinygrad import Tensor, dtypes
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tinygrad import nn
from typing import List, Callable
from rich import print
from rich.logging import RichHandler
import logging
import cv2

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
        load_state_dict(self, weights)


def load_and_preprocess_image(image_path: str) -> Tensor:
    """Load and preprocess a local image for MNIST prediction."""
    # Step 1: Load the image in grayscale mode (single channel, like MNIST)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Make sure the image was loaded successfully
    if img is None:
        log.error("Could Not Load Image")
        raise ValueError("Could Not Load Image")

    # Check if image is already in MNIST format (28x28 pixels)
    is_mnist = img.shape[0] == 28 and img.shape[1] == 28

    if not is_mnist:
        # Step 2: Resize and preprocess non-MNIST images
        # Start with a larger size for better quality processing
        target_size = 128
        aspect_ratio = img.shape[1] / img.shape[0]
        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Step 3: Enhance image contrast to separate digit from background
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Step 4: Convert to pure black and white using adaptive thresholding
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 5: Find and crop to the digit's bounding box
        coords = cv2.findNonZero(255 - img)
        if coords is not None:
            # Get the rectangle containing the digit
            x, y, w, h = cv2.boundingRect(coords)
            # Add 20% padding around the digit
            padding = int(max(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            img = img[y : y + h, x : x + w]

        # Step 6: Make the image square by adding black padding
        size = max(img.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_y = (size - img.shape[0]) // 2
        offset_x = (size - img.shape[1]) // 2
        square[
            offset_y : offset_y + img.shape[0], offset_x : offset_x + img.shape[1]
        ] = img

        # Step 7: Resize to MNIST size (28x28 pixels)
        img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 8: Convert to float32 (keep 0-255 range like MNIST)
    img = img.astype(np.float32)

    # Step 9: Invert if needed (MNIST uses white digits on black background)
    if np.mean(img) > 127:
        img = 255 - img

    # Step 10: Reshape to match MNIST format: (batch_size=1, channels=1, height=28, width=28)
    img = img.reshape(1, 1, 28, 28)

    # Step 11: Convert to TinyGrad tensor for model input
    return Tensor(img)


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
    img_array = load_and_preprocess_image(img_bytes)
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
    img_array = load_and_preprocess_image(image_path)
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
    # context = TensorContext()
    # input_image = context.add_graph_input("input_image", (1, 1, 28, 28), dtypes.float32)
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
