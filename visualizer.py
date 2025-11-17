"""
Модуль для визуализации результатов совмещения
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AlignmentVisualizer:
    """Создание визуализаций для проверки качества совмещения"""

    @staticmethod
    def draw_matches(img1: np.ndarray,
                     kp1: List,
                     img2: np.ndarray,
                     kp2: List,
                     matches: List,
                     mask: Optional[np.ndarray] = None,
                     max_display: int = 100) -> np.ndarray:
        """
        Рисует совпадающие точки между изображениями

        Args:
            img1, img2: изображения
            kp1, kp2: ключевые точки
            matches: список совпадений
            mask: маска inliers
            max_display: максимальное количество отображаемых линий

        Returns:
            изображение с нарисованными совпадениями
        """
        # Выбираем только inliers если есть маска
        if mask is not None:
            matches_to_draw = [m for i, m in enumerate(matches) if mask[i]]
        else:
            matches_to_draw = matches

        # Ограничиваем количество для читаемости
        if len(matches_to_draw) > max_display:
            matches_to_draw = matches_to_draw[:max_display]

        # Создаем цветные версии если нужно
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()

        if len(img2.shape) == 2:
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = img2.copy()

        # Рисуем совпадения
        match_img = cv2.drawMatches(
            img1_color, kp1,
            img2_color, kp2,
            matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return match_img

    @staticmethod
    def create_before_after(reference: np.ndarray,
                            target: np.ndarray,
                            aligned: np.ndarray) -> np.ndarray:
        """
        Создает сравнительную визуализацию до/после

        Args:
            reference: эталонное изображение
            target: исходное изображение второй камеры
            aligned: совмещенное изображение

        Returns:
            объединенное изображение с 3 панелями
        """

        # Конвертируем в цветное если нужно
        def to_color(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        ref_color = to_color(reference)
        tgt_color = to_color(target)
        aligned_color = to_color(aligned)

        # Добавляем подписи
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)

        cv2.putText(ref_color, "Reference (Cam1)", (10, 40),
                    font, font_scale, color, thickness)
        cv2.putText(tgt_color, "Original (Cam2)", (10, 40),
                    font, font_scale, color, thickness)
        cv2.putText(aligned_color, "Aligned (Cam2)", (10, 40),
                    font, font_scale, color, thickness)

        # Объединяем по вертикали
        combined = np.vstack([ref_color, tgt_color, aligned_color])

        return combined

    @staticmethod
    def create_difference_heatmap(reference: np.ndarray,
                                  aligned: np.ndarray,
                                  colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Создает тепловую карту разницы между изображениями

        Args:
            reference: эталонное изображение
            aligned: совмещенное изображение
            colormap: цветовая карта OpenCV

        Returns:
            тепловая карта разницы
        """
        # Конвертируем в grayscale
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference

        if len(aligned.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        else:
            aligned_gray = aligned

        # Вычисляем абсолютную разницу
        diff = cv2.absdiff(ref_gray, aligned_gray)

        # Нормализуем для лучшей визуализации
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        # Применяем цветовую карту
        heatmap = cv2.applyColorMap(diff_normalized.astype(np.uint8), colormap)

        # Добавляем статистику на изображение
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(heatmap, f"Mean diff: {mean_diff:.2f}", (10, 30),
                    font, 0.7, (255, 255, 255), 2)
        cv2.putText(heatmap, f"Max diff: {max_diff:.2f}", (10, 60),
                    font, 0.7, (255, 255, 255), 2)

        return heatmap

    @staticmethod
    def create_overlay(reference: np.ndarray,
                       aligned: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """
        Создает наложение двух изображений с прозрачностью

        Args:
            reference: эталонное изображение (зеленый канал)
            aligned: совмещенное изображение (красный канал)
            alpha: коэффициент прозрачности

        Returns:
            наложенное изображение
        """
        # Конвертируем в grayscale
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference

        if len(aligned.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        else:
            aligned_gray = aligned

        # Создаем цветное изображение: reference=зеленый, aligned=красный
        overlay = np.zeros((ref_gray.shape[0], ref_gray.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 1] = ref_gray  # Зеленый канал
        overlay[:, :, 2] = aligned_gray  # Красный канал

        # Смешиваем с оригиналами
        ref_color = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(ref_color, 1 - alpha, overlay, alpha, 0)

        return result

    @staticmethod
    def create_checkerboard(img1: np.ndarray,
                            img2: np.ndarray,
                            square_size: int = 50) -> np.ndarray:
        """
        Создает шахматное наложение для проверки совмещения

        Args:
            img1, img2: изображения для сравнения
            square_size: размер квадратов в пикселях

        Returns:
            шахматное изображение
        """
        # Конвертируем в цветное если нужно
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        h, w = img1.shape[:2]
        result = img1.copy()

        # Создаем шахматную маску
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    i_end = min(i + square_size, h)
                    j_end = min(j + square_size, w)
                    result[i:i_end, j:j_end] = img2[i:i_end, j:j_end]

        return result

    @staticmethod
    def create_comprehensive_report(reference: np.ndarray,
                                    target: np.ndarray,
                                    aligned: np.ndarray,
                                    results: dict) -> np.ndarray:
        """
        Создает комплексный отчет с несколькими визуализациями

        Args:
            reference: эталонное изображение
            target: исходное изображение
            aligned: совмещенное изображение
            results: результаты совмещения

        Returns:
            объединенное изображение с визуализациями
        """
        # Создаем отдельные визуализации
        overlay = AlignmentVisualizer.create_overlay(reference, aligned)
        heatmap = AlignmentVisualizer.create_difference_heatmap(reference, aligned)
        checkerboard = AlignmentVisualizer.create_checkerboard(reference, aligned)

        # Масштабируем для удобного отображения
        scale = 0.5
        overlay_small = cv2.resize(overlay, None, fx=scale, fy=scale)
        heatmap_small = cv2.resize(heatmap, None, fx=scale, fy=scale)
        checker_small = cv2.resize(checkerboard, None, fx=scale, fy=scale)

        # Верхний ряд
        top_row = np.hstack([overlay_small, heatmap_small])

        # Нижний ряд - checkerboard + текстовая информация
        info_img = np.zeros_like(checker_small)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        text_lines = [
            f"Keypoints: {results.get('keypoints_ref', 0)}, {results.get('keypoints_target', 0)}",
            f"Matches: {results.get('matches', 0)}",
            f"Inliers: {results.get('inliers', 0)} ({results.get('inlier_ratio', 0):.1%})",
            f"RMSE: {results.get('rmse', 0):.3f} px"
        ]

        for i, line in enumerate(text_lines):
            cv2.putText(info_img, line, (10, y_offset + i * 40),
                        font, 0.6, (255, 255, 255), 1)

        bottom_row = np.hstack([checker_small, info_img])

        # Объединяем
        report = np.vstack([top_row, bottom_row])

        return report
