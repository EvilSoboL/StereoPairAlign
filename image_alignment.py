"""
Модуль для совмещения изображений с двух камер
Использует feature-based alignment с ORB/SIFT
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageAligner:
    """Класс для совмещения изображений через гомографию"""

    def __init__(self,
                 feature_detector: str = 'orb',
                 max_features: int = 5000,
                 ransac_threshold: float = 3.0,
                 ransac_confidence: float = 0.995):
        """
        Args:
            feature_detector: 'orb', 'sift', или 'akaze'
            max_features: максимальное количество ключевых точек
            ransac_threshold: порог для RANSAC (пиксели)
            ransac_confidence: уровень уверенности RANSAC
        """
        self.feature_detector = feature_detector.lower()
        self.max_features = max_features
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence

        self.detector = self._init_detector()
        self.matcher = self._init_matcher()

    def _init_detector(self):
        """Инициализация детектора особенностей"""
        if self.feature_detector == 'sift':
            try:
                return cv2.SIFT_create(nfeatures=self.max_features)
            except AttributeError:
                logger.warning("SIFT недоступен, переключаюсь на ORB")
                self.feature_detector = 'orb'

        if self.feature_detector == 'orb':
            return cv2.ORB_create(nfeatures=self.max_features)

        if self.feature_detector == 'akaze':
            return cv2.AKAZE_create()

        raise ValueError(f"Неизвестный детектор: {self.feature_detector}")

    def _init_matcher(self):
        """Инициализация matcher'а"""
        if self.feature_detector in ['sift', 'akaze']:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # ORB
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Находит ключевые точки и дескрипторы

        Args:
            image: входное изображение

        Returns:
            keypoints, descriptors
        """
        # Конвертируем в grayscale если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        keypoints, descriptors = self.detector.detectAndCompute(enhanced, None)

        logger.info(f"Найдено {len(keypoints)} ключевых точек")

        return keypoints, descriptors

    def match_features(self,
                       desc1: np.ndarray,
                       desc2: np.ndarray,
                       ratio_threshold: float = 0.75) -> List:
        """
        Сопоставляет дескрипторы с Lowe's ratio test

        Args:
            desc1, desc2: дескрипторы изображений
            ratio_threshold: порог для фильтрации (обычно 0.7-0.8)

        Returns:
            список хороших совпадений
        """
        if desc1 is None or desc2 is None:
            logger.error("Дескрипторы пусты!")
            return []

        # Находим 2 ближайших соседа для каждого дескриптора
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Применяем Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        logger.info(f"Найдено {len(good_matches)} хороших совпадений из {len(matches)}")

        return good_matches

    def estimate_homography(self,
                            kp1: List,
                            kp2: List,
                            matches: List) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Вычисляет гомографию через RANSAC

        Args:
            kp1, kp2: ключевые точки обоих изображений
            matches: совпадения

        Returns:
            homography_matrix (3x3), mask (inliers)
        """
        if len(matches) < 4:
            logger.error("Недостаточно совпадений для гомографии (минимум 4)")
            return None, None

        # Извлекаем координаты точек
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Вычисляем гомографию
        H, mask = cv2.findHomography(
            pts2, pts1,
            cv2.RANSAC,
            self.ransac_threshold,
            confidence=self.ransac_confidence,
            maxIters=5000
        )

        if H is None:
            logger.error("Не удалось вычислить гомографию")
            return None, None

        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches)

        logger.info(f"Inliers: {inliers}/{len(matches)} ({inlier_ratio:.2%})")

        return H, mask

    def compute_rmse(self,
                     kp1: List,
                     kp2: List,
                     matches: List,
                     H: np.ndarray,
                     mask: np.ndarray) -> float:
        """
        Вычисляет RMSE для inliers

        Args:
            kp1, kp2: ключевые точки
            matches: совпадения
            H: матрица гомографии
            mask: inliers mask

        Returns:
            RMSE в пикселях
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Используем только inliers
        inlier_mask = mask.ravel() == 1
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]

        # Трансформируем точки второго изображения
        pts2_transformed = cv2.perspectiveTransform(
            pts2_inliers.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        # Вычисляем ошибки
        errors = np.linalg.norm(pts1_inliers - pts2_transformed, axis=1)
        rmse = np.sqrt(np.mean(errors ** 2))

        logger.info(f"RMSE: {rmse:.3f} пикселей")

        return float(rmse)

    def align_images(self,
                     reference_img: np.ndarray,
                     target_img: np.ndarray) -> Dict:
        """
        Полный процесс совмещения изображений

        Args:
            reference_img: эталонное изображение (нечетное)
            target_img: изображение для совмещения (четное)

        Returns:
            словарь с результатами
        """
        results = {
            'success': False,
            'homography': None,
            'keypoints_ref': 0,
            'keypoints_target': 0,
            'matches': 0,
            'inliers': 0,
            'inlier_ratio': 0.0,
            'rmse': None
        }

        # 1. Детекция ключевых точек
        kp1, desc1 = self.detect_and_compute(reference_img)
        kp2, desc2 = self.detect_and_compute(target_img)

        results['keypoints_ref'] = len(kp1)
        results['keypoints_target'] = len(kp2)

        if len(kp1) < 10 or len(kp2) < 10:
            logger.error("Слишком мало ключевых точек")
            return results

        # 2. Сопоставление
        matches = self.match_features(desc1, desc2)
        results['matches'] = len(matches)

        if len(matches) < 4:
            logger.error("Недостаточно совпадений")
            return results

        # 3. Вычисление гомографии
        H, mask = self.estimate_homography(kp1, kp2, matches)

        if H is None:
            return results

        results['homography'] = H
        results['inliers'] = int(np.sum(mask))
        results['inlier_ratio'] = results['inliers'] / len(matches)

        # 4. RMSE
        rmse = self.compute_rmse(kp1, kp2, matches, H, mask)
        results['rmse'] = rmse

        results['success'] = True
        results['_kp1'] = kp1  # Для визуализации
        results['_kp2'] = kp2
        results['_matches'] = matches
        results['_mask'] = mask

        return results

    def apply_homography(self,
                         image: np.ndarray,
                         H: np.ndarray,
                         output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Применяет гомографию к изображению

        Args:
            image: входное изображение
            H: матрица гомографии
            output_shape: (height, width) выходного изображения

        Returns:
            трансформированное изображение
        """
        return cv2.warpPerspective(image, H,
                                   (output_shape[1], output_shape[0]),
                                   flags=cv2.INTER_LINEAR)
