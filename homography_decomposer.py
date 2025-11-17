"""
Модуль для декомпозиции матрицы гомографии
на отдельные геометрические компоненты
"""

import numpy as np
import cv2
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class HomographyDecomposer:
    """Декомпозиция гомографии на rotation, scale, translation, perspective"""

    @staticmethod
    def decompose(H: np.ndarray) -> Dict:
        """
        Разлагает матрицу гомографии на компоненты

        Матрица гомографии имеет вид:
        H = [[h11, h12, h13],
             [h21, h22, h23],
             [h31, h32, h33]]

        Можно представить как:
        H = T * A * P
        где T - перспектива, A - аффинное преобразование

        Args:
            H: матрица гомографии 3x3

        Returns:
            словарь с компонентами преобразования
        """
        # Нормализуем матрицу
        H_normalized = H / H[2, 2]

        result = {
            'rotation_deg': 0.0,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'shift_x_px': 0.0,
            'shift_y_px': 0.0,
            'shear': 0.0,
            'perspective_x': 0.0,
            'perspective_y': 0.0,
            'perspective': [0.0, 0.0]
        }

        try:
            # Извлекаем компоненты
            h = H_normalized

            # 1. Translation (сдвиг)
            result['shift_x_px'] = float(h[0, 2])
            result['shift_y_px'] = float(h[1, 2])

            # 2. Perspective components (перспектива)
            result['perspective_x'] = float(h[2, 0])
            result['perspective_y'] = float(h[2, 1])
            result['perspective'] = [result['perspective_x'], result['perspective_y']]

            # 3. Извлекаем аффинную часть (верхний левый блок 2x2)
            A = h[:2, :2]

            # Разложение через SVD для получения rotation и scale
            # A = U * S * Vt = R1 * S * R2
            U, S, Vt = np.linalg.svd(A)

            # Rotation (поворот)
            R = U @ Vt

            # Проверяем, что это действительно поворот (det = 1)
            if np.linalg.det(R) < 0:
                # Если det = -1, это отражение, корректируем
                Vt[-1, :] *= -1
                R = U @ Vt
                S[-1] *= -1

            # Угол поворота
            rotation_rad = np.arctan2(R[1, 0], R[0, 0])
            result['rotation_deg'] = float(np.degrees(rotation_rad))

            # Scale (масштаб)
            result['scale_x'] = float(S[0])
            result['scale_y'] = float(S[1])

            # Shear (сдвиг/наклон) - можно вычислить через разложение
            # но для простоты используем соотношение компонент
            shear_rad = np.arctan2(-R[0, 1], R[0, 0]) - rotation_rad
            result['shear'] = float(np.degrees(shear_rad))

            logger.info(f"Декомпозиция: rotation={result['rotation_deg']:.2f}°, "
                       f"scale=({result['scale_x']:.3f}, {result['scale_y']:.3f}), "
                       f"shift=({result['shift_x_px']:.1f}, {result['shift_y_px']:.1f})")

        except Exception as e:
            logger.error(f"Ошибка при декомпозиции: {e}")

        return result

    @staticmethod
    def decompose_detailed(H: np.ndarray) -> Dict:
        """
        Более детальная декомпозиция с дополнительными метриками

        Args:
            H: матрица гомографии 3x3

        Returns:
            расширенный словарь параметров
        """
        basic = HomographyDecomposer.decompose(H)

        H_norm = H / H[2, 2]

        # Дополнительные метрики (все значения приводим к JSON-совместимым типам)
        detailed = {
            **basic,
            'is_pure_translation': False,
            'is_pure_rotation': False,
            'is_affine': False,
            'anisotropy': 1.0,
            'aspect_ratio_change': 1.0,
            'condition_number': 1.0
        }

        # Проверка на чисто аффинное преобразование
        perspective_magnitude = np.sqrt(H_norm[2, 0]**2 + H_norm[2, 1]**2)
        detailed['is_affine'] = bool(perspective_magnitude < 1e-6)

        # Проверка на чистый сдвиг
        A = H_norm[:2, :2]
        is_identity = np.allclose(A, np.eye(2), atol=1e-3)
        detailed['is_pure_translation'] = bool(is_identity)

        # Проверка на чистый поворот (без масштаба)
        scale_diff = abs(detailed['scale_x'] - 1.0) + abs(detailed['scale_y'] - 1.0)
        detailed['is_pure_rotation'] = bool(scale_diff < 0.01 and not detailed['is_pure_translation'])

        # Анизотропия (различие масштабов по осям)
        if detailed['scale_y'] != 0:
            detailed['anisotropy'] = float(detailed['scale_x'] / detailed['scale_y'])
            detailed['aspect_ratio_change'] = float(detailed['anisotropy'])

        # Число обусловленности (насколько хорошо определена матрица)
        try:
            _, S, _ = np.linalg.svd(H_norm[:2, :2])
            if S[-1] > 1e-10:
                detailed['condition_number'] = float(S[0] / S[-1])
        except:
            pass

        return detailed

    @staticmethod
    def reconstruct_from_components(rotation_deg: float,
                                   scale_x: float,
                                   scale_y: float,
                                   shift_x: float,
                                   shift_y: float,
                                   perspective_x: float = 0.0,
                                   perspective_y: float = 0.0) -> np.ndarray:
        """
        Восстанавливает матрицу гомографии из компонент
        (полезно для тестирования)

        Args:
            rotation_deg: угол поворота в градусах
            scale_x, scale_y: масштабы по осям
            shift_x, shift_y: сдвиги
            perspective_x, perspective_y: перспективные компоненты

        Returns:
            матрица гомографии 3x3
        """
        theta = np.radians(rotation_deg)
        c, s = np.cos(theta), np.sin(theta)

        # Rotation matrix
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # Scale matrix
        S = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ])

        # Translation matrix
        T = np.array([
            [1, 0, shift_x],
            [0, 1, shift_y],
            [0, 0, 1]
        ])

        # Affine part
        H_affine = T @ R @ S

        # Add perspective
        H_affine[2, 0] = perspective_x
        H_affine[2, 1] = perspective_y

        return H_affine