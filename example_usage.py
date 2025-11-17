"""
Примеры использования модулей для совмещения изображений
Демонстрирует программный API (без CLI)
"""

import cv2
import numpy as np
import json
from pathlib import Path

from image_alignment import ImageAligner
from homography_decomposer import HomographyDecomposer
from visualizer import AlignmentVisualizer


def example_1_basic_alignment():
    """
    Пример 1: Базовое совмещение двух изображений
    """
    print("=" * 70)
    print("ПРИМЕР 1: Базовое совмещение")
    print("=" * 70)

    # Загружаем изображения
    ref_img = cv2.imread('image7_b.bmp')
    tgt_img = cv2.imread('image8_b.bmp')

    # Создаем алгоритм совмещения
    aligner = ImageAligner(feature_detector='orb')

    # Выполняем совмещение
    results = aligner.align_images(ref_img, tgt_img)

    if results['success']:
        print(f"✅ Совмещение успешно!")
        print(f"   RMSE: {results['rmse']:.3f} пикселей")
        print(f"   Inliers: {results['inliers']}/{results['matches']}")

        # Применяем преобразование
        H = results['homography']
        aligned = aligner.apply_homography(tgt_img, H, ref_img.shape[:2])

        # Сохраняем результат
        cv2.imwrite('aligned_result.jpg', aligned)
    else:
        print("❌ Совмещение не удалось")


def example_2_decompose_homography():
    """
    Пример 2: Декомпозиция матрицы гомографии
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 2: Декомпозиция гомографии")
    print("=" * 70)

    # Пример матрицы гомографии
    H = np.array([
        [0.998, -0.012, 12.5],
        [0.015, 0.997, -5.3],
        [0.0001, -0.0002, 1.0]
    ])

    # Декомпозиция
    decomposer = HomographyDecomposer()
    components = decomposer.decompose_detailed(H)

    print("\nКомпоненты преобразования:")
    print(f"  Поворот:    {components['rotation_deg']:.3f}°")
    print(f"  Масштаб X:  {components['scale_x']:.6f}")
    print(f"  Масштаб Y:  {components['scale_y']:.6f}")
    print(f"  Сдвиг X:    {components['shift_x_px']:.2f} px")
    print(f"  Сдвиг Y:    {components['shift_y_px']:.2f} px")
    print(f"  Перспектива: {components['perspective']}")

    print(f"\nХарактеристики:")
    print(f"  Аффинное преобразование: {components['is_affine']}")
    print(f"  Анизотропия: {components['anisotropy']:.6f}")


def example_3_create_visualizations():
    """
    Пример 3: Создание визуализаций
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 3: Визуализации")
    print("=" * 70)

    # Загружаем изображения
    ref_img = cv2.imread('image7_b.bmp')
    tgt_img = cv2.imread('image8_b.bmp')

    # Совмещение
    aligner = ImageAligner()
    results = aligner.align_images(ref_img, tgt_img)

    if not results['success']:
        return

    aligned = aligner.apply_homography(tgt_img, results['homography'], ref_img.shape[:2])

    # Создаем визуализации
    visualizer = AlignmentVisualizer()

    # 1. Тепловая карта разницы
    heatmap = visualizer.create_difference_heatmap(ref_img, aligned)
    cv2.imwrite('viz_heatmap.jpg', heatmap)
    print("✅ Heatmap сохранена: viz_heatmap.jpg")

    # 2. Наложение с прозрачностью
    overlay = visualizer.create_overlay(ref_img, aligned, alpha=0.5)
    cv2.imwrite('viz_overlay.jpg', overlay)
    print("✅ Overlay сохранен: viz_overlay.jpg")

    # 3. Шахматное наложение
    checker = visualizer.create_checkerboard(ref_img, aligned, square_size=100)
    cv2.imwrite('viz_checkerboard.jpg', checker)
    print("✅ Checkerboard сохранен: viz_checkerboard.jpg")

    # 4. До/После
    before_after = visualizer.create_before_after(ref_img, tgt_img, aligned)
    cv2.imwrite('viz_before_after.jpg', before_after)
    print("✅ Before/After сохранен: viz_before_after.jpg")


def example_4_batch_processing():
    """
    Пример 4: Пакетная обработка множества пар
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 4: Пакетная обработка")
    print("=" * 70)

    from batch_processor import BatchProcessor

    # Создаем процессор
    processor = BatchProcessor(
        input_dir='data/images',
        output_dir='./results',
        feature_detector='orb',
        create_visualizations=True
    )

    # Обрабатываем все пары
    results = processor.process_all()

    print(f"\n✅ Обработано {len(results)} пар")

    # Анализ результатов
    if results:
        rmse_values = [r['quality']['rmse_pixels'] for r in results]
        print(f"   Средняя RMSE: {np.mean(rmse_values):.3f} px")
        print(f"   Мин RMSE: {np.min(rmse_values):.3f} px")
        print(f"   Макс RMSE: {np.max(rmse_values):.3f} px")


def example_5_custom_parameters():
    """
    Пример 5: Настройка параметров алгоритма
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 5: Кастомные параметры")
    print("=" * 70)

    # Создаем алгоритм с кастомными параметрами
    aligner = ImageAligner(
        feature_detector='orb',
        max_features=8000,  # Больше ключевых точек
        ransac_threshold=2.0,  # Более строгий порог
        ransac_confidence=0.999  # Выше уверенность
    )

    # Загружаем изображения
    ref_img = cv2.imread('image7_b.bmp')
    tgt_img = cv2.imread('image8_b.bmp')

    # Совмещение
    results = aligner.align_images(ref_img, tgt_img)

    print(f"Найдено keypoints: {results['keypoints_ref']}, {results['keypoints_target']}")
    print(f"Совпадений: {results['matches']}")
    print(f"Inliers: {results['inliers']} ({results['inlier_ratio']:.1%})")
    print(f"RMSE: {results['rmse']:.4f} px")


def example_6_save_load_homography():
    """
    Пример 6: Сохранение и загрузка гомографии
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 6: Сохранение/загрузка гомографии")
    print("=" * 70)

    # Вычисляем гомографию
    ref_img = cv2.imread('image7_b.bmp')
    tgt_img = cv2.imread('image8_b.bmp')

    aligner = ImageAligner()
    results = aligner.align_images(ref_img, tgt_img)

    if not results['success']:
        return

    H = results['homography']

    # Сохраняем в JSON
    output = {
        'homography': H.tolist(),
        'metadata': {
            'rmse': results['rmse'],
            'inliers': results['inliers']
        }
    }

    with open('homography.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("✅ Гомография сохранена в homography.json")

    # Загружаем обратно
    with open('homography.json', 'r') as f:
        loaded = json.load(f)

    H_loaded = np.array(loaded['homography'])

    # Применяем загруженную гомографию
    aligned = aligner.apply_homography(tgt_img, H_loaded, ref_img.shape[:2])
    cv2.imwrite('aligned_from_json.jpg', aligned)

    print("✅ Преобразование применено из загруженной матрицы")


def example_7_compare_detectors():
    """
    Пример 7: Сравнение разных детекторов
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 7: Сравнение детекторов")
    print("=" * 70)

    ref_img = cv2.imread('image7_b.bmp')
    tgt_img = cv2.imread('image8_b.bmp')

    detectors = ['orb', 'akaze']

    for detector in detectors:
        print(f"\n{detector.upper()}:")

        try:
            aligner = ImageAligner(feature_detector=detector)
            results = aligner.align_images(ref_img, tgt_img)

            if results['success']:
                print(f"  ✅ Keypoints: {results['keypoints_ref']}, {results['keypoints_target']}")
                print(f"  ✅ Matches: {results['matches']}")
                print(f"  ✅ Inliers: {results['inliers']} ({results['inlier_ratio']:.1%})")
                print(f"  ✅ RMSE: {results['rmse']:.4f} px")
            else:
                print(f"  ❌ Совмещение не удалось")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")


def example_8_synthetic_test():
    """
    Пример 8: Тест на синтетических данных
    """
    print("\n" + "=" * 70)
    print("ПРИМЕР 8: Синтетический тест")
    print("=" * 70)

    # Создаем синтетическое изображение
    height, width = 512, 512
    ref_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Добавляем некоторые структуры для детекции
    cv2.rectangle(ref_img, (100, 100), (400, 400), (255, 255, 255), 2)
    cv2.circle(ref_img, (256, 256), 50, (0, 255, 0), -1)

    # Создаем известное преобразование
    decomposer = HomographyDecomposer()
    H_known = decomposer.reconstruct_from_components(
        rotation_deg=5.0,
        scale_x=1.02,
        scale_y=0.98,
        shift_x=10.5,
        shift_y=-7.3
    )

    # Применяем трансформацию
    tgt_img = cv2.warpPerspective(ref_img, H_known, (width, height))

    # Пытаемся восстановить преобразование
    aligner = ImageAligner(feature_detector='orb')
    results = aligner.align_images(ref_img, tgt_img)

    if results['success']:
        H_recovered = results['homography']
        decomposed = decomposer.decompose(H_recovered)

        print("\nИзвестное преобразование:")
        print(f"  Поворот: 5.0°, Масштаб: (1.02, 0.98), Сдвиг: (10.5, -7.3)")

        print("\nВосстановленное преобразование:")
        print(f"  Поворот: {decomposed['rotation_deg']:.2f}°")
        print(f"  Масштаб: ({decomposed['scale_x']:.4f}, {decomposed['scale_y']:.4f})")
        print(f"  Сдвиг: ({decomposed['shift_x_px']:.2f}, {decomposed['shift_y_px']:.2f})")

        print(f"\n✅ RMSE: {results['rmse']:.4f} px")


def main():
    """Запуск всех примеров"""
    print("\n" + "🎯 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ API СОВМЕЩЕНИЯ ИЗОБРАЖЕНИЙ\n")

    examples = [
        ("Базовое совмещение", example_1_basic_alignment),
        ("Декомпозиция гомографии", example_2_decompose_homography),
        ("Визуализации", example_3_create_visualizations),
        ("Пакетная обработка", example_4_batch_processing),
        ("Кастомные параметры", example_5_custom_parameters),
        ("Сохранение/загрузка", example_6_save_load_homography),
        ("Сравнение детекторов", example_7_compare_detectors),
        ("Синтетический тест", example_8_synthetic_test),
    ]

    print("Доступные примеры:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nЗапустите отдельные функции из этого файла:")
    print("  >>> from example_usage import *")
    print("  >>> example_1_basic_alignment()")

    # Можно раскомментировать для запуска всех примеров
    # for name, func in examples:
    #     try:
    #         func()
    #     except Exception as e:
    #         print(f"❌ Ошибка в примере '{name}': {e}")


if __name__ == '__main__':
    main()
