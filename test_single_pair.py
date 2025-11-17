#!/usr/bin/env python3
"""
Скрипт для быстрого тестирования совмещения одной пары изображений
Полезен для отладки и подбора параметров

Использование:
    python test_single_pair.py ref.bmp target.bmp
    python test_single_pair.py ref.bmp target.bmp --detector sift --show
"""

import argparse
import cv2
import json
import logging
import sys
from pathlib import Path

from image_alignment import ImageAligner
from homography_decomposer import HomographyDecomposer
from visualizer import AlignmentVisualizer


def setup_logging(verbose: bool = False):
    """Настройка логирования"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args():
    """Парсинг аргументов"""
    parser = argparse.ArgumentParser(
        description='Тестирование совмещения одной пары изображений'
    )

    parser.add_argument(
        'reference',
        type=str,
        help='Путь к эталонному изображению (камера 1)'
    )

    parser.add_argument(
        'target',
        type=str,
        help='Путь к изображению для совмещения (камера 2)'
    )

    parser.add_argument(
        '--detector',
        type=str,
        choices=['orb', 'sift', 'akaze'],
        default='orb',
        help='Тип детектора особенностей'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Показать результаты в окнах OpenCV'
    )

    parser.add_argument(
        '--save',
        type=str,
        help='Сохранить результаты в директорию'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Детальное логирование'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Максимум ключевых точек'
    )

    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=3.0,
        help='Порог RANSAC'
    )

    return parser.parse_args()


def display_results(window_name: str, image: cv2.Mat, max_size: int = 1200):
    """
    Отображает изображение в окне с автоматическим масштабированием

    Args:
        window_name: имя окна
        image: изображение для отображения
        max_size: максимальный размер по большей стороне
    """
    h, w = image.shape[:2]

    # Масштабируем если нужно
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    cv2.imshow(window_name, image)


def main():
    """Главная функция"""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Проверка путей
    ref_path = Path(args.reference)
    tgt_path = Path(args.target)

    if not ref_path.exists():
        logger.error(f"Файл не найден: {args.reference}")
        sys.exit(1)

    if not tgt_path.exists():
        logger.error(f"Файл не найден: {args.target}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("ТЕСТ СОВМЕЩЕНИЯ ОДНОЙ ПАРЫ")
    logger.info("=" * 70)
    logger.info(f"Reference: {ref_path.name}")
    logger.info(f"Target:    {tgt_path.name}")
    logger.info(f"Detector:  {args.detector.upper()}")
    logger.info("=" * 70)

    # Загрузка изображений
    logger.info("\n1. Загрузка изображений...")
    ref_img = cv2.imread(str(ref_path))
    tgt_img = cv2.imread(str(tgt_path))

    if ref_img is None or tgt_img is None:
        logger.error("Ошибка загрузки изображений")
        sys.exit(1)

    logger.info(f"   Reference: {ref_img.shape}")
    logger.info(f"   Target:    {tgt_img.shape}")

    # Инициализация алгоритма
    logger.info(f"\n2. Инициализация {args.detector.upper()} детектора...")
    aligner = ImageAligner(
        feature_detector=args.detector,
        max_features=args.max_features,
        ransac_threshold=args.ransac_threshold
    )

    # Совмещение
    logger.info("\n3. Выполнение совмещения...")
    results = aligner.align_images(ref_img, tgt_img)

    if not results['success']:
        logger.error("❌ Совмещение не удалось!")
        sys.exit(1)

    logger.info("✅ Совмещение выполнено успешно!")

    # Применяем преобразование
    logger.info("\n4. Применение преобразования...")
    H = results['homography']
    aligned_img = aligner.apply_homography(tgt_img, H, ref_img.shape[:2])

    # Декомпозиция
    logger.info("\n5. Декомпозиция гомографии...")
    decomposer = HomographyDecomposer()
    decomposed = decomposer.decompose_detailed(H)

    # Вывод результатов
    logger.info("\n" + "=" * 70)
    logger.info("РЕЗУЛЬТАТЫ СОВМЕЩЕНИЯ")
    logger.info("=" * 70)

    logger.info("\nКачество:")
    logger.info(f"  Keypoints (ref):  {results['keypoints_ref']}")
    logger.info(f"  Keypoints (tgt):  {results['keypoints_target']}")
    logger.info(f"  Matches:          {results['matches']}")
    logger.info(f"  Inliers:          {results['inliers']} ({results['inlier_ratio']:.1%})")
    logger.info(f"  RMSE:             {results['rmse']:.4f} пикселей")

    logger.info("\nГеометрические параметры:")
    logger.info(f"  Поворот:          {decomposed['rotation_deg']:.3f}°")
    logger.info(f"  Масштаб X:        {decomposed['scale_x']:.6f}")
    logger.info(f"  Масштаб Y:        {decomposed['scale_y']:.6f}")
    logger.info(f"  Сдвиг X:          {decomposed['shift_x_px']:.2f} px")
    logger.info(f"  Сдвиг Y:          {decomposed['shift_y_px']:.2f} px")
    logger.info(f"  Перспектива X:    {decomposed['perspective_x']:.6f}")
    logger.info(f"  Перспектива Y:    {decomposed['perspective_y']:.6f}")

    logger.info("\nХарактеристики:")
    logger.info(f"  Анизотропия:      {decomposed['anisotropy']:.6f}")
    logger.info(f"  Число обуслов.:   {decomposed['condition_number']:.4f}")
    logger.info(f"  Аффинное:         {'Да' if decomposed['is_affine'] else 'Нет'}")

    # Оценка качества
    logger.info("\nОценка точности:")
    if results['rmse'] <= 1.0:
        logger.info("  ✅ Отличная точность (≤ 1 px)")
    elif results['rmse'] <= 2.0:
        logger.info("  ✓ Хорошая точность (≤ 2 px)")
    else:
        logger.warning("  ⚠ Низкая точность (> 2 px)")

    if results['inlier_ratio'] >= 0.8:
        logger.info("  ✅ Высокая согласованность (≥ 80%)")
    elif results['inlier_ratio'] >= 0.5:
        logger.info("  ✓ Средняя согласованность (≥ 50%)")
    else:
        logger.warning("  ⚠ Низкая согласованность (< 50%)")

    # Сохранение
    if args.save:
        logger.info(f"\n6. Сохранение результатов в {args.save}...")
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        output_data = {
            'reference': str(ref_path),
            'target': str(tgt_path),
            'homography': H.tolist(),
            'decomposed': decomposed,
            'quality': {
                'keypoints_ref': results['keypoints_ref'],
                'keypoints_target': results['keypoints_target'],
                'matches': results['matches'],
                'inliers': results['inliers'],
                'inlier_ratio': results['inlier_ratio'],
                'rmse': results['rmse']
            }
        }

        json_path = save_dir / 'result.json'
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"   JSON: {json_path}")

        # Визуализации
        viz = AlignmentVisualizer()

        # Before/After
        before_after = viz.create_before_after(ref_img, tgt_img, aligned_img)
        cv2.imwrite(str(save_dir / 'before_after.jpg'), before_after)

        # Heatmap
        heatmap = viz.create_difference_heatmap(ref_img, aligned_img)
        cv2.imwrite(str(save_dir / 'heatmap.jpg'), heatmap)

        # Overlay
        overlay = viz.create_overlay(ref_img, aligned_img)
        cv2.imwrite(str(save_dir / 'overlay.jpg'), overlay)

        # Matches
        if '_kp1' in results:
            matches_img = viz.draw_matches(
                ref_img, results['_kp1'],
                tgt_img, results['_kp2'],
                results['_matches'], results['_mask']
            )
            cv2.imwrite(str(save_dir / 'matches.jpg'), matches_img)

        # Checkerboard
        checker = viz.create_checkerboard(ref_img, aligned_img)
        cv2.imwrite(str(save_dir / 'checkerboard.jpg'), checker)

        logger.info("   Визуализации сохранены")

    # Отображение
    if args.show:
        logger.info("\n7. Отображение результатов (нажмите любую клавишу для закрытия)...")

        viz = AlignmentVisualizer()

        # Создаем визуализации
        overlay = viz.create_overlay(ref_img, aligned_img)
        heatmap = viz.create_difference_heatmap(ref_img, aligned_img)
        checker = viz.create_checkerboard(ref_img, aligned_img)

        if '_kp1' in results:
            matches_img = viz.draw_matches(
                ref_img, results['_kp1'],
                tgt_img, results['_kp2'],
                results['_matches'], results['_mask']
            )
            display_results("Matches", matches_img)

        display_results("Overlay", overlay)
        display_results("Heatmap", heatmap)
        display_results("Checkerboard", checker)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info("\n" + "=" * 70)
    logger.info("ТЕСТ ЗАВЕРШЕН")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
