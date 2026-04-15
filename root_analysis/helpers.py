from pathlib import Path

import cv2

def ensure_grayscale(im):
    """
    Convert an image to grayscale if it is not already.

    Parameters
    ----------
    im : numpy.ndarray
        Input image array. Can be either a 2D grayscale image or a 3D color 
        image with 3 channels (BGR/BGRA format).

    Returns
    -------
    numpy.ndarray
        Grayscale image as a 2D array.

    Raises
    ------
    ValueError
        If the input image has an unexpected shape (neither 2D nor 3D with 3 channels).

    Examples
    --------
    >>> gray_img = ensure_grayscale(color_image)
    >>> gray_img = ensure_grayscale(already_gray_image)  # Returns unchanged
    """
    if len(im.shape) == 3 and im.shape[2] == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    elif len(im.shape) == 2:
        gray = im
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return gray

def crop_and_save(bbox, image_path, output_path, show=False, dpi=100):
    """Crop and save a square region from an image.

    Parameters
    ----------
    bbox : tuple
        ((x1, y1), (x2, y2)) Coordinates of the top-left and bottom-right
        corners of the crop rectangle.
    image_path : str or pathlib.Path
        Path to the source image.
    output_path : str or pathlib.Path
        Path where the cropped image will be saved.
    show : bool, optional
        If True, display the cropped image.
    dpi : int, optional
        DPI to use when displaying the image.

    Returns
    -------
    pathlib.Path
        Path to the saved cropped image.

    Raises
    ------
    ValueError
        If bbox is not a valid rectangle (e.g. zero or negative area).
    """

    im = cv2.imread(str(image_path))

    (x1, y1), (x2, y2) = bbox

    cropped = im[y1:y2, x1:x2]

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), cropped)

    if show:
        plt.figure(dpi=dpi)
        plt.imshow(cropped)
        plt.title(str(output_path))
        plt.show()
    return output_path


def get_mask(img_path, thresh=128, maxval=255, show=False, dpi=300):
    """Return a binary mask for an image using Otsu's thresholding.

    Parameters
    ----------
    img_path : str or pathlib.Path
        Path to the input image.
    thresh : int, optional
        Initial threshold value (ignored when using Otsu's method).
    maxval : int, optional
        Value to assign to pixels greater than the threshold.
    show : bool, optional
        If True, display the resulting mask.
    dpi : int, optional
        DPI to use when displaying the image.

    Returns
    -------
    ret : float
        Computed threshold value (from Otsu's method when requested).
    binary_mask : numpy.ndarray
        Binary mask image (dtype uint8) with values in {0, maxval}.
    """

    im = cv2.imread(img_path)
    im = ensure_grayscale(im)
    ret, binary_mask = cv2.threshold(im, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show:
        plt.figure(dpi=dpi)
        plt.imshow(binary_mask, cmap='gray')
        plt.show()
    return ret, binary_mask


# +
def get_bounding_box(binary_mask):
    """Compute a square bounding box around the largest contour in a binary mask.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        2D binary image (dtype uint8) where foreground pixels are non-zero (e.g. 255).

    Returns
    -------
    tuple
        A pair of 2-tuples ((x1, y1), (x2, y2)) giving the top-left and
        bottom-right coordinates of a square bounding box that encloses the
        largest contour. Coordinates are clamped to the image boundaries (the
        function uses the module-level `im` array for image shape).

    Raises
    ------
    ValueError
        If no contours are found in the provided mask.
    """
    contours, hiearchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the biggest contour
    max_contour = max(contours, key=cv2.contourArea)

    # get the x, y, width, height of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # get the longest side
    size = max(w, h)

    # find the center of each side
    center_x = x + w // 2
    center_y = y + h // 2

    # calculate the new x, y based on the longest side
    new_x = center_x - size // 2
    new_y = center_y - size // 2

    # make sure it is inside the borders of the image
    new_x = min(new_x, binary_mask.shape[1] - size)
    new_y = min(new_y, binary_mask.shape[0] - size)

    return((new_x, new_y), (new_x + size, new_y + size))