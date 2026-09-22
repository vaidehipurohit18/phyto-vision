from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def validate_image_file(file):
    """
    Validate an uploaded image file.

    Returns:
        is_valid (bool)
        error_message (str or None)
        pil_image (PIL.Image or None)
    """

    if file is None:
        return False, "No image file uploaded.", None

    if not file.filename:
        return False, "No file selected.", None

    # Check extension
    if "." not in file.filename:
        return False, "Invalid file format. Please upload JPG, JPEG, or PNG.", None

    extension = file.filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return False, "Only JPG, JPEG, and PNG images are allowed.", None

    try:
        # Read and verify image
        image = Image.open(file.stream)
        image.verify()

        # Reset stream after verification
        file.stream.seek(0)

        # Reopen image
        image = Image.open(file.stream).convert("RGB")

        return True, None, image

    except (UnidentifiedImageError, OSError, ValueError):
        return False, "Invalid or corrupted image file.", None