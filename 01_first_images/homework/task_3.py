import cv2
import numpy as np


def rotate(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    while angle > 90:
        image = rotate(image, point, 90)
        angle -= 90

    h, w, _ = image.shape

    angle_sin = np.sin(np.deg2rad(angle))
    angle_cos = np.cos(np.deg2rad(angle))

    new_h = int(h * angle_cos + w * angle_sin)
    new_w = int(w * angle_cos + h * angle_sin)

    pts1 = np.float32([[0, 0], [0, h], [w, 0]])
    pts2 = np.float32(
        [[0, w * angle_sin], [h * angle_sin, new_h], [new_w - h * angle_sin, 0]]
    )

    M = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(image, M, (new_w, new_h))


def scan_notebook(image: np.ndarray) -> np.ndarray:
    """
    Сканирует тетрадь на изображении

    :param image: исходное изображение
    :return: преобразованное изображение
    """
    h, w, _ = image.shape
    image_copy_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    red_hsv_low = (0, 0, 0)
    red_hsv_high = (6, 255, 255)

    notebook_area = cv2.inRange(image_copy_hsv, red_hsv_low, red_hsv_high)

    corners = [None, None, None, None]
    new_corners = np.float32([(0, h), (0, 0), (w, 0), (w, h)])

    for i in range(w):
        for j in range(h):
            if corners[0] and corners[2]:
                break
            # Проверка достижение нижнего левого угла
            if notebook_area[j][i] != 0:
                corners[0] = (i, j)
            # Проверка достижение правого верхнего угла
            if notebook_area[j][w - i - 1] != 0:
                corners[2] = (w - i - 1, j)

    for i in range(h):
        for j in range(w):
            if corners[1] and corners[3]:
                break
            # Проверка достижение верхнего левого угла
            if notebook_area[i][j] != 0:
                corners[1] = (j, i)
            # Проверка достижение правого нижнего угла
            if notebook_area[h - i - 1][j] != 0:
                corners[3] = (j, h - i - 1)

    M = cv2.getPerspectiveTransform(np.float32(corners), new_corners)

    return cv2.warpPerspective(image, M, (w, h))
