"""
Unit тесты для системы совмещения изображений
Запуск: pytest test_alignment.py -v
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from image_alignment import ImageAligner
from homography_decomposer import HomographyDecomposer
from visualizer import AlignmentVisualizer


class TestHomographyDecomposer:
    """Тесты для декомпозиции гомографии"""

    def test_identity_matrix(self):
        """Тест на единичную матрицу (без преобразований)"""
        H = np.eye(3)
        decomposer = HomographyDecomposer()
        result = decomposer.decompose(H)

        assert abs(result['rotation_deg']) < 0.01
        assert abs(result['scale_x'] - 1.0) < 0.01
        assert abs(result['scale_y'] - 1.0) < 0.01
        assert abs(result['shift_x_px']) < 0.01
        assert abs(result['shift_y_px']) < 0.01

    def test_pure_translation(self):
        """Тест на чистый сдвиг"""
        H = np.array([
            [1, 0, 10],
            [0, 1, 20],
            [0, 0, 1]
        ])

        decomposer = HomographyDecomposer()
        result = decomposer.decompose(H)

        assert abs(result['shift_x_px'] - 10) < 0.01
        assert abs(result['shift_y_px'] - 20) < 0.01
        assert abs(result['rotation_deg']) < 0.01
        assert abs(result['scale_x'] - 1.0) < 0.01

    def test_pure_rotation(self):
        """Тест на чистый поворот"""
        angle_deg = 30
        angle_rad = np.radians(angle_deg)

        H = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        decomposer = HomographyDecomposer()
        result = decomposer.decompose(H)

        assert abs(result['rotation_deg'] - angle_deg) < 0.1
        assert abs(result['scale_x'] - 1.0) < 0.01

    def test_pure_scale(self):
        """Тест на чистое масштабирование"""
        H = np.array([
            [1.5, 0, 0],
            [0, 2.0, 0],
            [0, 0, 1]
        ])

        decomposer = HomographyDecomposer()
        result = decomposer.decompose(H)

        assert abs(result['scale_x'] - 1.5) < 0.01
        assert abs(result['scale_y'] - 2.0) < 0.01
        assert abs(result['rotation_deg']) < 0.01

    def test_reconstruction(self):
        """Тест на восстановление матрицы из компонент"""
        decomposer = HomographyDecomposer()

        # Создаем матрицу
        H_original = decomposer.reconstruct_from_components(
            rotation_deg=15.0,
            scale_x=1.1,
            scale_y=0.9,
            shift_x=5.0,
            shift_y=-3.0
        )

        # Декомпозируем
        components = decomposer.decompose(H_original)

        # Проверяем
        assert abs(components['rotation_deg'] - 15.0) < 0.1
        assert abs(components['scale_x'] - 1.1) < 0.01
        assert abs(components['scale_y'] - 0.9) < 0.01
        assert abs(components['shift_x_px'] - 5.0) < 0.1
        assert abs(components['shift_y_px'] + 3.0) < 0.1


class TestImageAligner:
    """Тесты для алгоритма совмещения"""

    @pytest.fixture
    def synthetic_image_pair(self):
        """Создает синтетическую пару изображений для тестов"""
        # Создаем изображение с геометрией
        size = 512
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Добавляем структуры
        cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), 2)
        cv2.circle(img, (256, 256), 80, (200, 200, 200), -1)
        cv2.line(img, (0, 256), (512, 256), (150, 150, 150), 2)
        cv2.line(img, (256, 0), (256, 512), (150, 150, 150), 2)

        # Добавляем случайный шум для текстуры
        noise = np.random.randint(0, 50, (size, size, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

        # Создаем трансформированную версию
        angle = 5
        scale = 1.02
        tx, ty = 10, -7

        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Преобразуем в гомографию 3x3
        H = np.vstack([M, [0, 0, 1]])

        img_transformed = cv2.warpPerspective(img, H, (size, size))

        return img, img_transformed, H

    def test_detector_initialization(self):
        """Тест инициализации детекторов"""
        for detector in ['orb', 'akaze']:
            aligner = ImageAligner(feature_detector=detector)
            assert aligner.detector is not None
            assert aligner.matcher is not None

    def test_keypoint_detection(self, synthetic_image_pair):
        """Тест детекции ключевых точек"""
        img, _, _ = synthetic_image_pair

        aligner = ImageAligner(feature_detector='orb')
        kp, desc = aligner.detect_and_compute(img)

        assert len(kp) > 0
        assert desc is not None
        assert desc.shape[0] == len(kp)

    def test_feature_matching(self, synthetic_image_pair):
        """Тест сопоставления особенностей"""
        img1, img2, _ = synthetic_image_pair

        aligner = ImageAligner(feature_detector='orb')
        kp1, desc1 = aligner.detect_and_compute(img1)
        kp2, desc2 = aligner.detect_and_compute(img2)

        matches = aligner.match_features(desc1, desc2)

        assert len(matches) > 0

    def test_alignment_accuracy(self, synthetic_image_pair):
        """Тест точности совмещения на синтетических данных"""
        img1, img2, H_true = synthetic_image_pair

        aligner = ImageAligner(feature_detector='orb', max_features=3000)
        results = aligner.align_images(img1, img2)

        assert results['success']
        assert results['rmse'] < 5.0  # Синтетические данные должны давать хорошую точность
        assert results['inlier_ratio'] > 0.5

    def test_homography_properties(self, synthetic_image_pair):
        """Тест свойств вычисленной гомографии"""
        img1, img2, _ = synthetic_image_pair

        aligner = ImageAligner(feature_detector='orb')
        results = aligner.align_images(img1, img2)

        if results['success']:
            H = results['homography']

            # Матрица должна быть 3x3
            assert H.shape == (3, 3)

            # Нормализованная: H[2,2] должна быть близка к 1
            H_norm = H / H[2, 2]
            assert abs(H_norm[2, 2] - 1.0) < 0.01

            # Должна быть обратимой
            det = np.linalg.det(H)
            assert abs(det) > 1e-6


class TestVisualizer:
    """Тесты для визуализатора"""

    @pytest.fixture
    def test_images(self):
        """Создает тестовые изображения"""
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return img1, img2

    def test_create_overlay(self, test_images):
        """Тест создания наложения"""
        img1, img2 = test_images

        visualizer = AlignmentVisualizer()
        overlay = visualizer.create_overlay(img1, img2, alpha=0.5)

        assert overlay.shape == img1.shape
        assert overlay.dtype == np.uint8

    def test_create_heatmap(self, test_images):
        """Тест создания тепловой карты"""
        img1, img2 = test_images

        visualizer = AlignmentVisualizer()
        heatmap = visualizer.create_difference_heatmap(img1, img2)

        assert heatmap.shape == (*img1.shape[:2], 3)
        assert heatmap.dtype == np.uint8

    def test_create_checkerboard(self, test_images):
        """Тест создания шахматного наложения"""
        img1, img2 = test_images

        visualizer = AlignmentVisualizer()
        checker = visualizer.create_checkerboard(img1, img2, square_size=50)

        assert checker.shape == img1.shape
        assert checker.dtype == np.uint8

    def test_before_after(self, test_images):
        """Тест создания сравнения до/после"""
        img1, img2 = test_images
        img3 = img2.copy()

        visualizer = AlignmentVisualizer()
        result = visualizer.create_before_after(img1, img2, img3)

        # Должно быть 3 изображения по вертикали
        assert result.shape[0] == img1.shape[0] * 3
        assert result.shape[1] == img1.shape[1]


class TestIntegration:
    """Интеграционные тесты полного процесса"""

    def test_full_pipeline_synthetic(self):
        """Тест полного процесса на синтетических данных"""
        # Создаем изображение
        size = 512
        img1 = np.random.randint(100, 200, (size, size, 3), dtype=np.uint8)
        cv2.circle(img1, (256, 256), 100, (255, 255, 255), -1)
        cv2.rectangle(img1, (150, 150), (350, 350), (200, 200, 200), 2)

        # Создаем известное преобразование
        decomposer = HomographyDecomposer()
        H_known = decomposer.reconstruct_from_components(
            rotation_deg=3.0,
            scale_x=1.01,
            scale_y=0.99,
            shift_x=8.0,
            shift_y=-5.0
        )

        img2 = cv2.warpPerspective(img1, H_known, (size, size))

        # Выполняем совмещение
        aligner = ImageAligner(feature_detector='orb')
        results = aligner.align_images(img1, img2)

        assert results['success'], "Совмещение должно быть успешным"

        # Декомпозируем
        H_recovered = results['homography']
        components = decomposer.decompose(H_recovered)

        # Проверяем точность восстановления
        assert abs(components['rotation_deg'] - 3.0) < 1.0
        assert abs(components['shift_x_px'] - 8.0) < 2.0
        assert abs(components['shift_y_px'] + 5.0) < 2.0

        # RMSE должна быть малой
        assert results['rmse'] < 3.0

    def test_edge_cases(self):
        """Тест граничных случаев"""
        aligner = ImageAligner()

        # Пустые изображения
        empty1 = np.zeros((100, 100, 3), dtype=np.uint8)
        empty2 = np.zeros((100, 100, 3), dtype=np.uint8)

        results = aligner.align_images(empty1, empty2)
        assert not results['success'], "Пустые изображения не должны совмещаться"

        # Одинаковые изображения
        same_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.circle(same_img, (128, 128), 50, (255, 255, 255), -1)

        results = aligner.align_images(same_img, same_img)
        if results['success']:
            # Если совмещение успешно, гомография должна быть близка к единичной
            H = results['homography']
            H_norm = H / H[2, 2]
            assert np.allclose(H_norm, np.eye(3), atol=0.1)


def test_imports():
    """Тест что все модули импортируются"""
    import image_alignment
    import homography_decomposer
    import visualizer
    import batch_processor

    assert True


if __name__ == '__main__':
    # Запуск тестов
    pytest.main([__file__, '-v', '--tb=short'])
