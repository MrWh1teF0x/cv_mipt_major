import cv2
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def find_way_from_maze(image: np.ndarray) -> list[tuple]:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    image_copy = image.copy()
    h, w, _ = image.shape

    start, end, size = find_entry_and_exit(image_copy)
    color_wall(image_copy)

    path = [start]
    path.append(down(start, size))

    while path[-1] != end:
        possible_moves = dict()
        possible_moves["up"] = up(path[-1], size)
        possible_moves["down"] = down(path[-1], size)
        possible_moves["right"] = right(path[-1], size)
        possible_moves["left"] = left(path[-1], size)

        if not (
            possible_moves["up"] != path[-2]
            and not is_wall(image_copy, possible_moves["up"])
        ):
            del possible_moves["up"]

        if not (
            possible_moves["down"] != path[-2]
            and not is_wall(image_copy, possible_moves["down"])
        ):
            del possible_moves["down"]

        if not (
            possible_moves["right"] != path[-2]
            and not is_wall(image_copy, possible_moves["right"])
        ):
            del possible_moves["right"]

        if not (
            possible_moves["left"] != path[-2]
            and not is_wall(image_copy, possible_moves["left"])
        ):
            del possible_moves["left"]

        if len(possible_moves) == 1:
            for move in possible_moves:
                path.append(possible_moves[move])
        else:
            for move in possible_moves:

                if move == "up" or move == "down":
                    left_wall_x, left_wall_y = left(possible_moves[move], size)
                    right_wall_x, right_wall_y = right(possible_moves[move], size)

                    color_left_wall = image_copy[left_wall_y][left_wall_x]
                    color_right_wall = image_copy[right_wall_y][right_wall_x]
                    if not (all(color_left_wall == color_right_wall)):
                        path.append(possible_moves[move])
                        break

                else:
                    up_wall_x, up_wall_y = up(possible_moves[move], size)
                    down_wall_x, down_wall_y = down(possible_moves[move], size)

                    color_up_wall = image_copy[up_wall_y][up_wall_x]
                    color_down_wall = image_copy[down_wall_y][down_wall_x]
                    if not (all(color_up_wall == color_down_wall)):
                        path.append(possible_moves[move])
                        break

    return path


def color_wall(image: np.ndarray) -> np.ndarray:
    """
    Красит левую стенку в красный цвет

    :param image: изображение лабиринта
    :return: копию изображения с покрашенной левой стенкой
    """
    h, w, _ = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    _, image, _, _ = cv2.floodFill(image, mask, (0, 0), RED)


def find_entry_and_exit(image: np.ndarray) -> tuple[tuple, tuple, int]:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты входа и выхода из лабиринта в виде (x, y), где x и y - это массивы координат, а также размер клеток, из которых состоит лабиринт
    """
    h, w, _ = image.shape
    start_start, start_end = None, None
    end_start, end_end = None, None

    # Находим вход
    for i in range(w):
        if start_start:
            if all(image[0][i] == BLACK):
                start_end = (i - 1, 0)
                break
        elif all(image[0][i] == WHITE):
            start_start = (i, 0)

    # Находим выход
    for i in range(w):
        if end_start:
            if all(image[h - 1][i] == BLACK):
                end_end = (i - 1, h - 2)
                break
        elif all(image[h - 1][i] == WHITE):
            end_start = (i, h - 2)

    start = (start_end[0] - ((start_end[0] - start_start[0]) // 2), 0)
    end = (end_end[0] - ((end_end[0] - end_start[0]) // 2), h - 2)
    size = start_end[0] - start_start[0] + 2

    return (start, end, size)


def up(pos: tuple[int, int], size) -> tuple[tuple, tuple]:
    """
    Движение вверх в лабиринте

    :param pos: текущая позиция пути
    :param size: размер клетки в лабиринте
    :return: координата после движения вверх в виде (x, y)
    """
    x, y = pos[0], pos[1]
    return (x, y - size // 2 - 1)


def down(pos: tuple[int, int], size) -> tuple[tuple, tuple]:
    """
    Движение вниз в лабиринте

    :param pos: текущая позиция пути
    :param size: размер клетки в лабиринте
    :return: координата после движения вниз в виде (x, y)
    """
    x, y = pos[0], pos[1]
    return (x, y + size // 2 + 1)


def right(pos: tuple[int, int], size) -> tuple[tuple, tuple]:
    """
    Движение вправо в лабиринте

    :param pos: текущая позиция пути
    :param size: размер клетки в лабиринте
    :return: координата после движения вправо в виде (x, y)
    """
    x, y = pos[0], pos[1]
    return (x + size // 2 + 1, y)


def left(pos: tuple[int, int], size) -> tuple[tuple, tuple]:
    """
    Движение влево в лабиринте

    :param pos: текущая позиция пути
    :param size: размер клетки в лабиринте
    :return: координата после движения влево в виде (x, y)
    """
    x, y = pos[0], pos[1]
    return (x - size // 2 - 1, y)


def is_wall(image: np.ndarray, curr_pos: tuple) -> bool:
    """
    Проверка на стену

    :param image: изображение лабиринта
    :param curr_pos: текущая позиция пути
    :return: является текущая точка частью стены
    """
    x, y = curr_pos[0], curr_pos[1]
    return True if all(image[y][x] == BLACK) or all(image[y][x] == RED) else False
