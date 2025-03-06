import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> np.ndarray:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    h, w, _ = image.shape
    image_copy_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    yellow_hsv_low = (20, 100, 100)
    yellow_hsv_high = (40, 255, 255)

    borders_area = cv2.inRange(image_copy_hsv, yellow_hsv_low, yellow_hsv_high)

    roads_range = []
    start_new_road = False
    start_road, end_road = None, None

    for i in range(w):
        if borders_area[0][i] == 0 and not start_new_road:
            start_new_road = True
            start_road = i
        elif borders_area[0][i] != 0 and start_new_road:
            start_new_road = False
            end_road = i
            roads_range.append((start_road, end_road))
            start_road, end_road = None, None

    red_hsv_low = (0, 100, 100)
    red_hsv_high = (10, 255, 255)

    obstacles_area = cv2.inRange(image_copy_hsv, red_hsv_low, red_hsv_high)

    free_road = None

    for i, road_range in enumerate(roads_range):
        road_min, road_max = road_range[0], road_range[1]
        road_average = road_max - (road_max - road_min) // 2

        for j in range(h):
            if obstacles_area[j][road_average] != 0:
                break
        else:
            free_road = i
            break

    blue_hsv_low = (100, 100, 100)
    blue_hsv_high = (140, 255, 255)

    car_area = cv2.inRange(image_copy_hsv, blue_hsv_low, blue_hsv_high)
    car_on_road_index = None

    for i, road_range in enumerate(roads_range):
        road_min, road_max = road_range[0], road_range[1]
        road_average = road_max - (road_max - road_min) // 2

        for j in range(h):
            if car_area[j][road_average] != 0:
                car_on_road_index = i
                break

    return free_road if car_on_road_index != free_road else None
