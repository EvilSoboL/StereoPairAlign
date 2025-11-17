#!/usr/bin/env python3
"""
Главный скрипт для совмещения изображений с двух камер
Использование:
    python main.py --input ./images --output ./results
    python main.py --input ./images --output ./results --detector sift --viz
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from batch_processor import BatchProcessor
except ImportError:
    print("ERROR: Не найден модуль batch_processor.py")
    print("Убедитесь, что все файлы находятся в одной директории:")
    print("  - image_alignment.py")
    print("  - homography_decomposer.py")
    print("  - visualizer.py")
    print("  - batch_processor.py")
    print("  - main.py")
    sys.exit(1)


def setup_logging(verbose: bool = False, log_file: str = None):
    """
    Настройка логирования

    Args:
        verbose: включить детальное логирование
        log_file: путь к файлу лога
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Формат лога
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Настройка корневого логгера
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Уменьшаем уровень для сторонних библиотек
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Автоматическое совмещение изображений с двух камер',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовая обработка с ORB детектором
  python main.py --input ./images --output ./results

  # Использование SIFT с визуализациями
  python main.py -i ./images -o ./results --detector sift --viz

  # Детальное логирование в файл
  python main.py -i ./images -o ./results -v --log processing.log

Схема нумерации файлов:
  - Нечетные номера (7, 9, 11, ...) - эталонные изображения (камера 1)
  - Четные номера (8, 10, 12, ...) - изображения для совмещения (камера 2)

Формат: image<NUM>_b.bmp
        """
    )

    # Обязательные аргументы
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Путь к директории с входными BMP изображениями'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Путь к директории для результатов (JSON + визуализации)'
    )

    # Опциональные параметры
    parser.add_argument(
        '--detector',
        type=str,
        choices=['orb', 'sift', 'akaze'],
        default='orb',
        help='Тип детектора особенностей (по умолчанию: orb)'
    )

    parser.add_argument(
        '--viz', '--visualizations',
        action='store_true',
        dest='visualizations',
        help='Создавать визуализации (до/после, heatmap, overlay)'
    )

    parser.add_argument(
        '--no-viz',
        action='store_false',
        dest='visualizations',
        help='Не создавать визуализации (только JSON)'
    )

    parser.set_defaults(visualizations=True)

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Детальное логирование'
    )

    parser.add_argument(
        '--log',
        type=str,
        help='Путь к файлу лога (опционально)'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Максимальное количество ключевых точек (по умолчанию: 5000)'
    )

    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=3.0,
        help='Порог RANSAC в пикселях (по умолчанию: 3.0)'
    )

    return parser.parse_args()


def validate_paths(input_dir: str, output_dir: str):
    """
    Валидация путей

    Args:
        input_dir: путь к входной директории
        output_dir: путь к выходной директории
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Входная директория не существует: {input_dir}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Путь не является директорией: {input_dir}")

    # Проверяем наличие BMP файлов
    bmp_files = list(input_path.glob('*.bmp'))
    if not bmp_files:
        raise FileNotFoundError(f"В директории {input_dir} не найдено BMP файлов")

    logging.info(f"Найдено {len(bmp_files)} BMP файлов в {input_dir}")


def main():
    """Главная функция"""
    args = parse_args()

    # Настраиваем логирование
    setup_logging(verbose=args.verbose, log_file=args.log)

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("СИСТЕМА СОВМЕЩЕНИЯ ИЗОБРАЖЕНИЙ С ДВУХ КАМЕР")
    logger.info("=" * 70)

    # Валидация путей
    try:
        validate_paths(args.input, args.output)
    except Exception as e:
        logger.error(f"Ошибка валидации: {e}")
        sys.exit(1)

    # Выводим конфигурацию
    logger.info("\nКонфигурация:")
    logger.info(f"  Входная директория:  {args.input}")
    logger.info(f"  Выходная директория: {args.output}")
    logger.info(f"  Детектор:           {args.detector.upper()}")
    logger.info(f"  Визуализации:       {'Да' if args.visualizations else 'Нет'}")
    logger.info(f"  Max features:       {args.max_features}")
    logger.info(f"  RANSAC threshold:   {args.ransac_threshold} px")

    # Создаем процессор
    try:
        processor = BatchProcessor(
            input_dir=args.input,
            output_dir=args.output,
            feature_detector=args.detector,
            create_visualizations=args.visualizations
        )

        # Применяем дополнительные параметры
        processor.aligner.max_features = args.max_features
        processor.aligner.ransac_threshold = args.ransac_threshold
        processor.aligner.detector = processor.aligner._init_detector()

    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        sys.exit(1)

    # Обрабатываем все пары
    logger.info("\nНачало обработки...\n")

    try:
        results = processor.process_all()

        if not results:
            logger.warning("\nНи одна пара не была успешно обработана")
            sys.exit(1)

        logger.info("\n" + "=" * 70)
        logger.info("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО")
        logger.info("=" * 70)
        logger.info(f"\nОбработано пар: {len(results)}")
        logger.info(f"Результаты сохранены в: {args.output}")

        # Статистика
        avg_rmse = sum(r['quality']['rmse_pixels'] for r in results) / len(results)
        avg_inliers = sum(r['quality']['inlier_ratio'] for r in results) / len(results)

        logger.info(f"\nСредняя точность совмещения:")
        logger.info(f"  RMSE: {avg_rmse:.3f} пикселей")
        logger.info(f"  Inlier ratio: {avg_inliers:.1%}")

        if avg_rmse <= 1.0:
            logger.info("\n✓ Отличная точность совмещения (≤ 1 пиксель)")
        elif avg_rmse <= 2.0:
            logger.info("\n✓ Хорошая точность совмещения (≤ 2 пикселя)")
        else:
            logger.warning(f"\n⚠ Точность совмещения ниже ожидаемой (> 2 пикселя)")

    except KeyboardInterrupt:
        logger.info("\n\nОбработка прервана пользователем")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nОшибка при обработке: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
