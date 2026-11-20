"""
Model Evaluation and Verification
Project: Waste Classification Using CNN

Purpose:
1. Check whether the ONNX model exists.
2. Check whether the ONNX model is valid.
3. Measure the model size.
4. Estimate the number of model parameters.
5. Check model input and output shapes.
6. Measure CPU inference latency.
7. Calculate validation accuracy.

Usage:

    python verify_submission.py --model model.onnx

    python verify_submission.py --model model.onnx --val_dir data/public/val
"""

import argparse
import os
import time

import numpy as np


# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 96

# The order must match the training code.
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Supported image formats.
# JFIF is included because some dataset images use this format.
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".webp",
)

# Normalization values used during training.
IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)


# --------------------------------------------------
# 2. DISPLAY EVALUATION RESULTS
# --------------------------------------------------

def show_result(name, detail):
    """
    Display one evaluation result.

    Parameters:
        name (str): Name of the evaluation metric.
        detail (str): Measured value or explanation.
    """

    print(f"{name:25}: {detail}")


# --------------------------------------------------
# 3. CALCULATE MODEL PARAMETERS
# --------------------------------------------------

def count_model_parameters(model_path):
    """
    Count the values stored in the ONNX model's
    weight tensors.

    Parameters:
        model_path (str): Path of the ONNX model.

    Returns:
        int: Number of stored parameter values.
    """

    import onnx

    # Load the ONNX model.
    model = onnx.load(model_path)

    # Check whether the model follows ONNX rules.
    onnx.checker.check_model(model)

    parameter_count = 0

    # Initializers contain stored model weights
    # and other constant tensors.
    for tensor in model.graph.initializer:

        parameter_count += int(
            np.prod(tensor.dims, dtype=np.int64)
        )

    return parameter_count


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def prepare_image(image_path):
    """
    Prepare an image before giving it to the ONNX model.

    Steps:
        1. Open the image.
        2. Convert it into RGB format.
        3. Resize it to 96 x 96 pixels.
        4. Convert pixel values into the range 0 to 1.
        5. Change image layout from HWC to CHW.
        6. Apply normalization.
        7. Add a batch dimension.

    Parameters:
        image_path (str): Path of the input image.

    Returns:
        numpy.ndarray: Preprocessed image with shape
                       (1, 3, 96, 96).
    """

    from PIL import Image

    # Open image and convert it to RGB.
    with Image.open(image_path) as image:

        image = image.convert("RGB")

        # Resize image.
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert image into a NumPy array.
        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    # Change layout from:
    # Height, Width, Channels
    # to:
    # Channels, Height, Width
    array = np.transpose(array, (2, 0, 1))

    # Normalize image using training values.
    array = (array - IMAGE_MEAN) / IMAGE_STD

    # Add batch dimension.
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


# --------------------------------------------------
# 5. VALIDATION ACCURACY
# --------------------------------------------------

def evaluate_validation(
    session,
    input_name,
    output_name,
    val_dir
):
    """
    Calculate model accuracy using validation images.

    Parameters:
        session: ONNX Runtime inference session.
        input_name (str): Name of model input.
        output_name (str): Name of model output.
        val_dir (str): Validation dataset folder.

    Returns:
        tuple:
            accuracy (float): Validation accuracy percentage.
            total (int): Number of evaluated images.
    """

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation folder not found: {val_dir}"
        )

    correct = 0
    total = 0

    # Visit each class in the expected class order.
    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(val_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}"
            )

        # Find supported images in the class folder.
        # Convert entries into a list and sort by filename.
        # Sorting DirEntry objects directly causes a TypeError.
        with os.scandir(class_dir) as entries:

            image_paths = sorted(
                (
                    path
                    for path in entries
                    if path.is_file()
                    and path.name.lower().endswith(IMAGE_EXTENSIONS)
                ),
                key=lambda path: path.name.lower()
            )

        # Evaluate each image.
        for image_entry in image_paths:

            image_input = prepare_image(image_entry.path)

            # Run prediction using ONNX Runtime.
            scores = session.run(
                [output_name],
                {input_name: image_input}
            )[0]

            # Select the class with the highest score.
            predicted_class = int(
                np.argmax(scores, axis=1)[0]
            )

            # Compare prediction with actual class.
            if predicted_class == class_index:
                correct += 1

            total += 1

    if total == 0:
        raise ValueError(
            "No supported validation images were found."
        )

    # Calculate accuracy percentage.
    accuracy = (correct / total) * 100.0

    return accuracy, total


# --------------------------------------------------
# 6. MODEL VERIFICATION AND EVALUATION
# --------------------------------------------------

def verify_model(model_path, val_dir=""):
    """
    Perform model checks and measure performance.

    Parameters:
        model_path (str): Path of the ONNX model.
        val_dir (str): Optional validation dataset path.

    Returns:
        int: 0 if evaluation completes,
             1 if an important check fails,
             2 if some evaluation is incomplete.
    """

    print("\n" + "=" * 60)
    print("WASTE CLASSIFICATION USING CNN")
    print("MODEL EVALUATION AND VERIFICATION")
    print("=" * 60)

    print(f"\nModel file: {model_path}")

    # Check whether the model file exists.
    if not os.path.isfile(model_path):

        print("\nERROR: ONNX model file not found.")
        return 1

    incomplete = False
    failed = False

    # --------------------------------------------------
    # A. MODEL SIZE
    # --------------------------------------------------

    print("\n1. MODEL SIZE")
    print("-" * 60)

    size_bytes = os.path.getsize(model_path)

    size_mb = size_bytes / (1024 * 1024)

    show_result(
        "ONNX model size",
        f"{size_mb:.3f} MB"
    )

    # --------------------------------------------------
    # B. MODEL PARAMETERS
    # --------------------------------------------------

    print("\n2. MODEL PARAMETERS")
    print("-" * 60)

    try:

        parameter_count = count_model_parameters(model_path)

        show_result(
            "Stored parameter values",
            f"{parameter_count:,}"
        )

    except ImportError:

        print("ONNX package is not installed.")
        print("Install it using: pip install onnx")

        incomplete = True

    except Exception as error:

        print(f"ONNX model check failed: {error}")

        failed = True

    # --------------------------------------------------
    # C. MODEL INPUT AND OUTPUT
    # --------------------------------------------------

    print("\n3. MODEL INPUT AND OUTPUT")
    print("-" * 60)

    session = None
    input_name = None
    output_name = None
    valid_signature = False

    try:

        import onnxruntime as ort

        # Create inference session using CPU.
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Expected one model input and one model output."
            )

        model_input = inputs[0]
        model_output = outputs[0]

        input_name = model_input.name
        output_name = model_output.name

        input_shape = model_input.shape
        output_shape = model_output.shape

        print(f"Input name  : {input_name}")
        print(f"Input shape : {input_shape}")
        print(f"Input type  : {model_input.type}")

        print(f"\nOutput name  : {output_name}")
        print(f"Output shape : {output_shape}")
        print(f"Output type  : {model_output.type}")

        # Check expected model format.
        valid_signature = (
            len(input_shape) == 4
            and input_shape[1:] == [
                3,
                IMAGE_SIZE,
                IMAGE_SIZE
            ]
            and len(output_shape) == 2
            and output_shape[1] == len(CLASS_NAMES)
            and model_input.type == "tensor(float)"
            and model_output.type == "tensor(float)"
        )

        if valid_signature:

            show_result(
                "Input/output format",
                "Correct: float32 (Batch, 3, 96, 96) -> (Batch, 6)"
            )

        else:

            show_result(
                "Input/output format",
                "Does not match the expected format."
            )

            failed = True

    except ImportError:

        print("ONNX Runtime is not installed.")
        print("Install it using: pip install onnxruntime")

        incomplete = True

    except Exception as error:

        print(f"Model runtime check failed: {error}")

        failed = True

    # --------------------------------------------------
    # D. CPU INFERENCE LATENCY
    # --------------------------------------------------

    print("\n4. CPU INFERENCE LATENCY")
    print("-" * 60)

    if session is not None and valid_signature:

        try:

            # Create a sample input image.
            sample = np.zeros(
                (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.float32
            )

            # Warm-up runs are performed before timing.
            for _ in range(5):

                session.run(
                    [output_name],
                    {input_name: sample}
                )

            timings = []

            # Measure 50 inference runs.
            for _ in range(50):

                start_time = time.perf_counter()

                session.run(
                    [output_name],
                    {input_name: sample}
                )

                end_time = time.perf_counter()

                elapsed_ms = (
                    end_time - start_time
                ) * 1000.0

                timings.append(elapsed_ms)

            average_latency = float(np.mean(timings))

            show_result(
                "Average CPU latency",
                f"{average_latency:.3f} ms/image"
            )

        except Exception as error:

            print(f"Latency measurement failed: {error}")

            failed = True

    else:

        print("Latency measurement skipped.")

        incomplete = True

    # --------------------------------------------------
    # E. VALIDATION ACCURACY
    # --------------------------------------------------

    print("\n5. VALIDATION ACCURACY")
    print("-" * 60)

    if val_dir:

        if session is not None and valid_signature:

            try:

                accuracy, sample_count = evaluate_validation(
                    session,
                    input_name,
                    output_name,
                    val_dir
                )

                show_result(
                    "Validation accuracy",
                    f"{accuracy:.2f}%"
                )

                show_result(
                    "Evaluated images",
                    str(sample_count)
                )

            except Exception as error:

                print(f"Validation evaluation failed: {error}")

                failed = True

        else:

            print("Validation accuracy could not be measured.")

            incomplete = True

    else:

        print("Validation folder was not provided.")
        print("Use --val_dir to calculate validation accuracy.")

        incomplete = True

    # --------------------------------------------------
    # 7. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)

    if failed:

        print("Some model checks failed. Review the errors.")

        return 1

    elif incomplete:

        print("Evaluation incomplete. Some measurements were skipped.")

        return 2

    else:

        print("Model evaluation completed successfully.")

        return 0


# --------------------------------------------------
# 8. COMMAND-LINE ARGUMENTS
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the ONNX waste classification model."
        )
    )

    parser.add_argument(
        "--model",
        default="model.onnx",
        help="Path to the ONNX model."
    )

    parser.add_argument(
        "--val_dir",
        default="",
        help="Path to the validation dataset."
    )

    args = parser.parse_args()

    return verify_model(
        args.model,
        args.val_dir
    )


# --------------------------------------------------
# 9. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    raise SystemExit(main())
