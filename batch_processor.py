"""
Модуль для пакетной обработки пар изображений
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

try:
    from image_alignment import ImageAligner
    from homography_decomposer import HomographyDecomposer
    from visualizer import AlignmentVisualizer
except ImportError as e:
    print(f"ERROR: Не удалось импортировать необходимые модули: {e}")
    print("\nУбедитесь, что в директории присутствуют файлы:")
    print("  - image_alignment.py")
    print("  - homography_decomposer.py")
    print("  - visualizer.py")
    raise

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Обработчик для пакетной обработки пар изображений"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 feature_detector: str = 'orb',
                 create_visualizations: bool = True):
        """
        Args:
            input_dir: директория с входными изображениями
            output_dir: директория для результатов
            feature_detector: тип детектора ('orb', 'sift', 'akaze')
            create_visualizations: создавать ли визуализации
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.create_visualizations = create_visualizations

        # Создаем поддиректории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if create_visualizations:
            (self.output_dir / 'visualizations').mkdir(exist_ok=True)

        self.aligner = ImageAligner(feature_detector=feature_detector)
        self.decomposer = HomographyDecomposer()
        self.visualizer = AlignmentVisualizer()

        self.results_log = []

    def find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Находит пары изображений по схеме нечетное-четное

        Returns:
            список кортежей (reference_path, target_path)
        """
        # Находим все BMP файлы
        all_images = sorted(self.input_dir.glob('*.bmp'))

        pairs = []

        # Извлекаем номера из имен файлов
        image_dict = {}
        for img_path in all_images:
            name = img_path.stem
            # Ожидаем формат image<NUM>_b.bmp
            try:
                # Извлекаем число из имени
                parts = name.split('_')
                if len(parts) >= 1:
                    num_str = ''.join(filter(str.isdigit, parts[0]))
                    if num_str:
                        num = int(num_str)
                        image_dict[num] = img_path
            except ValueError:
                logger.warning(f"Не удалось распарсить имя файла: {name}")
                continue

        # Создаем пары: нечетное (reference) + четное (target)
        sorted_nums = sorted(image_dict.keys())

        for i in range(len(sorted_nums) - 1):
            num1 = sorted_nums[i]
            num2 = sorted_nums[i + 1]

            # Проверяем, что это последовательные числа и нечетное-четное
            if num2 == num1 + 1 and num1 % 2 == 1:
                pairs.append((image_dict[num1], image_dict[num2]))
                logger.info(f"Найдена пара: {image_dict[num1].name} <-> {image_dict[num2].name}")

        logger.info(f"Всего найдено {len(pairs)} пар изображений")
        return pairs

    def process_pair(self,
                     reference_path: Path,
                     target_path: Path) -> Optional[Dict]:
        """
        Обрабатывает одну пару изображений

        Args:
            reference_path: путь к эталонному изображению
            target_path: путь к изображению для совмещения

        Returns:
            словарь с результатами или None при ошибке
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Обработка пары:")
        logger.info(f"  Reference: {reference_path.name}")
        logger.info(f"  Target:    {target_path.name}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # Загружаем изображения
        try:
            reference_img = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            target_img = cv2.imread(str(target_path), cv2.IMREAD_COLOR)

            if reference_img is None or target_img is None:
                logger.error("Не удалось загрузить изображения")
                return None

            logger.info(f"Размер изображений: {reference_img.shape}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке: {e}")
            return None

        # Выполняем совмещение
        align_results = self.aligner.align_images(reference_img, target_img)

        if not align_results['success']:
            logger.error("Совмещение не удалось")
            return None

        # Применяем преобразование
        H = align_results['homography']
        aligned_img = self.aligner.apply_homography(
            target_img, H, reference_img.shape[:2]
        )

        # Декомпозиция гомографии
        decomposed = self.decomposer.decompose_detailed(H)

        # Формируем результат
        processing_time = time.time() - start_time

        result = {
            'reference_file': reference_path.name,
            'target_file': target_path.name,
            'processing_time_sec': round(processing_time, 3),
            'homography': H.tolist(),
            'transform_decomposed': decomposed,
            'quality': {
                'keypoints_reference': align_results['keypoints_ref'],
                'keypoints_target': align_results['keypoints_target'],
                'matched_points': align_results['matches'],
                'inliers': align_results['inliers'],
                'inlier_ratio': round(align_results['inlier_ratio'], 4),
                'rmse_pixels': round(align_results['rmse'], 4)
            }
        }

        # Сохраняем JSON
        output_name = target_path.stem
        json_path = self.output_dir / f"{output_name}_transform.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Сохранен JSON: {json_path.name}")

        # Визуализации
        if self.create_visualizations:
            self._create_visualizations(
                reference_img, target_img, aligned_img,
                align_results, output_name
            )

        # Логируем результат
        logger.info(f"\nРезультаты совмещения:")
        logger.info(
            f"  Keypoints: {result['quality']['keypoints_reference']} / {result['quality']['keypoints_target']}")
        logger.info(f"  Matches: {result['quality']['matched_points']}")
        logger.info(f"  Inliers: {result['quality']['inliers']} ({result['quality']['inlier_ratio']:.1%})")
        logger.info(f"  RMSE: {result['quality']['rmse_pixels']:.3f} px")
        logger.info(f"  Rotation: {decomposed['rotation_deg']:.2f}°")
        logger.info(f"  Scale: ({decomposed['scale_x']:.4f}, {decomposed['scale_y']:.4f})")
        logger.info(f"  Shift: ({decomposed['shift_x_px']:.1f}, {decomposed['shift_y_px']:.1f}) px")
        logger.info(f"  Processing time: {processing_time:.2f} sec")

        self.results_log.append(result)

        return result

    def _create_visualizations(self,
                               reference_img: np.ndarray,
                               target_img: np.ndarray,
                               aligned_img: np.ndarray,
                               align_results: Dict,
                               output_name: str):
        """Создает и сохраняет визуализации"""
        viz_dir = self.output_dir / 'visualizations'

        try:
            # 1. Сравнение до/после
            before_after = self.visualizer.create_before_after(
                reference_img, target_img, aligned_img
            )
            cv2.imwrite(str(viz_dir / f"{output_name}_before_after.jpg"), before_after)

            # 2. Тепловая карта разницы
            heatmap = self.visualizer.create_difference_heatmap(
                reference_img, aligned_img
            )
            cv2.imwrite(str(viz_dir / f"{output_name}_heatmap.jpg"), heatmap)

            # 3. Наложение
            overlay = self.visualizer.create_overlay(reference_img, aligned_img)
            cv2.imwrite(str(viz_dir / f"{output_name}_overlay.jpg"), overlay)

            # 4. Совпадающие точки
            if '_kp1' in align_results and '_matches' in align_results:
                matches_img = self.visualizer.draw_matches(
                    reference_img,
                    align_results['_kp1'],
                    target_img,
                    align_results['_kp2'],
                    align_results['_matches'],
                    align_results['_mask']
                )
                cv2.imwrite(str(viz_dir / f"{output_name}_matches.jpg"), matches_img)

            # 5. Шахматное наложение
            checkerboard = self.visualizer.create_checkerboard(
                reference_img, aligned_img, square_size=100
            )
            cv2.imwrite(str(viz_dir / f"{output_name}_checkerboard.jpg"), checkerboard)

            logger.info(f"Визуализации сохранены в {viz_dir}")

        except Exception as e:
            logger.error(f"Ошибка при создании визуализаций: {e}")

    def process_all(self) -> List[Dict]:
        """
        Обрабатывает все пары изображений в директории

        Returns:
            список результатов для всех пар
        """
        pairs = self.find_image_pairs()

        if not pairs:
            logger.warning("Не найдено пар изображений для обработки")
            return []

        results = []

        for ref_path, tgt_path in pairs:
            try:
                result = self.process_pair(ref_path, tgt_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при обработке пары {ref_path.name} - {tgt_path.name}: {e}")
                continue

        # Сохраняем сводный отчет
        self._save_summary_report(results)

        return results

    def _save_summary_report(self, results: List[Dict]):
        """Сохраняет сводный отчет по всем парам"""
        if not results:
            return

        summary_path = self.output_dir / 'summary_report.json'

        summary = {
            'total_pairs': len(results),
            'successful': len([r for r in results if r is not None]),
            'average_rmse': np.mean([r['quality']['rmse_pixels'] for r in results]),
            'average_inlier_ratio': np.mean([r['quality']['inlier_ratio'] for r in results]),
            'total_processing_time': sum(r['processing_time_sec'] for r in results),
            'pairs': results
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\nСводный отчет сохранен: {summary_path}")
        logger.info(f"Обработано пар: {summary['successful']}/{summary['total_pairs']}")
        logger.info(f"Средняя RMSE: {summary['average_rmse']:.3f} px")
        logger.info(f"Средняя доля inliers: {summary['average_inlier_ratio']:.1%}")
