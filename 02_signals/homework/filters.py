import numpy as np


def conv_nested(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel)

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for hi in range(Hi):
        for wi in range(Wi):
            s = 0
            for hk in range(Hk):
                for wk in range(Wk):
                    image_h = hi - 1 + hk
                    image_w = wi - 1 + wk
                    if (0 <= image_h <= Hi - 1) and (0 <= image_w <= Wi - 1):
                        s += image[image_h][image_w] * kernel[hk][wk]

            out[hi][wi] = s

    return out


def zero_pad(image: np.ndarray, pad_height: int, pad_width: int) -> np.ndarray:
    """Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    Hi, Wi = image.shape

    new_image = np.ndarray((Hi + 2 * pad_height, Wi + 2 * pad_width))

    for h in range(pad_height, pad_height + Hi):
        for w in range(pad_width, pad_width + Wi):
            new_image[h][w] = image[h - pad_height][w - pad_width]

    return new_image


def conv_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel)

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = zero_pad(image, Hk // 2, Wk // 2)

    for hi in range(Hi):
        for wi in range(Wi):
            M = image[hi : hi + Hk, wi : wi + Wk]
            out[hi][wi] = np.sum(M * kernel)

    return out


def quarter_roll(img: np.ndarray, new_h: int, num_w: int) -> np.ndarray:
    """Pads the image and rolls the images quarters so that they get into corners

    Args:
        img: numpy array of shape (Hi, Wi).
        new_h: height of result
        new_w: width if result

    Returns:
        img_rolled: numpy array of shape (new_h, new_w).
    """
    Hi, Wi = img.shape
    roll_height = Hi // 2
    roll_width = Wi // 2

    img_rolled = np.zeros((new_h, num_w), dtype=img.dtype)
    img_rolled[:Hi, :Wi] = img
    img_rolled = np.roll(img_rolled, shift=(-roll_height, -roll_width), axis=(0, 1))
    return img_rolled


def conv_faster(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    padded_image = np.zeros([max(Hi, Wi), max(Hi, Wi)], dtype=image.dtype)
    padded_image[:Hi, :Wi] = image
    padded_kernel = quarter_roll(kernel, max(Hi, Wi), max(Hi, Wi))

    fourier_image = np.fft.ifft2(padded_image)
    fourier_kernel = np.fft.ifft2(padded_kernel)
    fourier_out = fourier_image * fourier_kernel

    return np.real(np.fft.fft2(fourier_out))[:Hi, :Wi]


def cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_fast(f, np.flip(g))


def zero_mean_cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    return conv_fast(f, np.flip(g - np.mean(g)))


def normalized_cross_correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))

    f = zero_pad(f, Hg // 2, Wg // 2)
    g = (g - np.mean(g)) / np.std(g)

    for hf in range(Hf):
        for wf in range(Wf):
            M = f[hf : hf + Hg, wf : wf + Wg]
            M = (M - np.mean(M)) / np.std(M)
            out[hf, wf] = np.sum(M * g)

    return out
