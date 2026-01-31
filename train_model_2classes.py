# train_model_2classes.py
import os
import yaml
import torch
from ultralytics import YOLO
import argparse


def train_yolo_model(data_yaml_path, epochs=30, img_size=640, batch_size=16, model_type='yolov8n.pt'):
    """
    Функция для дообучения модели YOLOv8 только на 2 классах: столы и люди

    Args:
        data_yaml_path (str): путь к файлу data.yaml с описанием датасета
        epochs (int): количество эпох обучения
        img_size (int): размер входного изображения
        batch_size (int): размер батча
        model_type (str): тип предобученной модели
    """

    # Загрузка конфигурации датасета
    with open(data_yaml_path, 'r', encoding='utf-8') as file:
        dataset_config = yaml.safe_load(file)

    print(f"  Загружена конфигурация датасета из {data_yaml_path}")
    print(f"  Количество классов: {dataset_config.get('nc', 'не указано')}")
    print(f"  Названия классов: {dataset_config.get('names', 'не указаны')}")

    # Проверка что у нас 2 класса
    num_classes = dataset_config.get('nc', 0)
    class_names = dataset_config.get('names', [])
    
    if num_classes != 2 or 'table' not in class_names or 'people' not in class_names:
        print("  Предупреждение: датасет должен содержать только 2 класса: 'table' и 'people'")
    
    # Загрузка предобученной модели
    print(f" Загрузка предобученной модели {model_type}")
    model = YOLO(model_type)

    # Настройка параметров обучения
    training_params = {
        'data': data_yaml_path,  # путь к конфигурации датасета
        'epochs': epochs,        # количество эпох
        'imgsz': img_size,       # размер изображения
        'batch': batch_size,     # размер батча
        'device': '0' if torch.cuda.is_available() else 'cpu',  # использование GPU если доступно
        'save_period': 10,       # сохранение чекпоинта каждые 10 эпох
        'cache': False,          # отключаем кэш для экономии памяти на CPU
        'name': 'cafe_table_people_detection',  # имя эксперимента
        'exist_ok': True,        # перезапись существующих результатов
        'project': 'runs/train'  # папка для сохранения результатов
    }

    # Дополнительные параметры для улучшения качества обучения
    training_params.update({
        'optimizer': 'AdamW',    # оптимизатор AdamW
        'lr0': 0.01,             # начальный learning rate (увеличил для быстрой сходимости)
        'lrf': 0.01,             # финальный learning rate
        'momentum': 0.937,       # momentum
        'weight_decay': 0.0005,  # weight decay
        'warmup_epochs': 3.0,    # эпохи прогрева
        'box': 7.5,              # вес лосса для bounding box
        'cls': 0.5,              # вес лосса для классификации
        'dfl': 1.5,              # вес лосса для distribution focal loss
        'hsv_h': 0.015,          # аугментация HSV Hue
        'hsv_s': 0.7,            # аугментация HSV Saturation
        'hsv_v': 0.4,            # аугментация HSV Value
        'degrees': 10.0,         # аугментация поворота ±10 градусов
        'translate': 0.1,        # аугментация сдвига
        'scale': 0.5,            # аугментация масштаба
        'fliplr': 0.5,           # аугментация горизонтального флипа
        'mosaic': 1.0,           # аугментация мозаики
        'mixup': 0.1,            # аугментация mixup для улучшения обобщения
        'copy_paste': 0.3        # copy-paste для людей за столами
    })

    print(f"  Параметры обучения:")
    print(f"   - Эпох: {epochs}")
    print(f"   - Размер изображения: {img_size}")
    print(f"   - Размер батча: {batch_size}")
    print(f"   - Модель: {model_type}")
    print(f"   - Устройство: {'GPU (CUDA)' if training_params['device'] != 'cpu' else 'CPU'}")
    print(f"   - Learning rate: {training_params['lr0']}")

    #  КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: СРАЗУ ЗАПУСКАЕМ ОБУЧЕНИЕ
    print("\n" + "="*70)
    print(" ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ...")
    print("="*70)
    print(" Процесс может занять 30-90 минут в зависимости от оборудования")
    print(" Прогресс обучения будет отображаться в реальном времени")
    print("="*70 + "\n")
    
    # ЗАПУСК ОБУЧЕНИЯ
    results = model.train(**training_params)
    
    # Вывод итоговых результатов
    print("\n" + "="*70)
    print(" ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*70)
    
    # Получаем метрики из результатов
    metrics = results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
    precision = metrics.get('metrics/precision(B)', 0)
    recall = metrics.get('metrics/recall(B)', 0)
    
    print(f" Результаты на валидационной выборке:")
    print(f"   - mAP50:    {map50:.4f}")
    print(f"   - mAP50-95: {map50_95:.4f}")
    print(f"   - Precision:{precision:.4f}")
    print(f"   - Recall:   {recall:.4f}")
    print(f"\n Лучшая модель сохранена в:")
    print(f"   {results.save_dir}/weights/best.pt")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Дообучение модели YOLOv8 для детекции ТОЛЬКО столов и людей в кафе')
    parser.add_argument('--data', type=str, default='dataset/data_2classes.yaml',
                        help='Путь к файлу data.yaml с описанием датасета (только 2 класса)')
    parser.add_argument('--epochs', type=int, default=50,  # Увеличил до 50 для лучшего результата
                        help='Количество эпох обучения (рекомендуется 50)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Размер входного изображения (по умолчанию 640)')
    parser.add_argument('--batch-size', type=int, default=8,  # Уменьшил до 8 для стабильной работы на CPU
                        help='Размер батча (рекомендуется 8 для CPU, 16+ для GPU)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Тип предобученной модели (по умолчанию yolov8n.pt)')

    args = parser.parse_args()

    # Проверка существования файла датасета
    if not os.path.exists(args.data):
        print(f" Файл датасета {args.data} не найден!")
        print("Убедитесь, что вы создали файл data_2classes.yaml с 2 классами")
        return

    print(" Запуск скрипта дообучения модели YOLOv8 (только 2 класса)")
    print(f" Параметры:")
    print(f"   - Датасет: {args.data}")
    print(f"   - Эпох: {args.epochs}")
    print(f"   - Размер изображения: {args.img_size}")
    print(f"   - Размер батча: {args.batch_size}")
    print(f"   - Модель: {args.model}")
    print("="*70)

    #  КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: СРАЗУ ЗАПУСКАЕМ ОБУЧЕНИЕ (без возврата параметров)
    try:
        results = train_yolo_model(
            data_yaml_path=args.data,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            model_type=args.model
        )
        print("\n Обучение успешно завершено! Модель готова к использованию.")
    except Exception as e:
        print(f"\n Критическая ошибка во время обучения: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n Советы по устранению ошибок:")
        print("   - Если ошибка памяти: уменьшите --batch-size до 4")
        print("   - Если ошибка с путями: проверьте структуру датасета")
        print("   - Для отладки запустите с --epochs 5 для быстрой проверки")


if __name__ == "__main__":
    main()