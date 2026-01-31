# filter_to_2_classes.py
import os
import yaml
from pathlib import Path
import shutil

def filter_annotations_for_2_classes(dataset_path, output_dataset_path):
    
    # Загружаем оригинальную конфигурацию датасета
    with open(os.path.join(dataset_path, 'data.yaml'), 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Определяем соответствие между старыми и новыми индексами классов
    original_names = original_config['names']
    target_classes = ['table', 'people']
    
    # Находим индексы целевых классов в оригинальном датасете
    new_to_old_mapping = {}
    old_to_new_mapping = {}
    
    for new_idx, target_class in enumerate(target_classes):
        if target_class in original_names:
            old_idx = original_names.index(target_class)
            new_to_old_mapping[new_idx] = old_idx
            old_to_new_mapping[old_idx] = new_idx
        else:
            print(f"Предупреждение: Класс '{target_class}' не найден в оригинальном датасете")
    
    print(f"Соответствие: {old_to_new_mapping}")
    
    # Создаем новую структуру датасета
    splits = ['train', 'valid', 'test']  # Измените в зависимости от ваших реальных разделов
    
    for split in splits:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue
            
        # Создаем выходные директории
        out_images_dir = os.path.join(output_dataset_path, split, 'images')
        out_labels_dir = os.path.join(output_dataset_path, split, 'labels')
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_labels_dir, exist_ok=True)
        
        # Обрабатываем каждый файл аннотаций
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                output_label_path = os.path.join(out_labels_dir, label_file)
                
                # Читаем оригинальную аннотацию
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Фильтруем и переназначаем аннотации
                filtered_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        
                        # Проверяем, нужно ли сохранить этот класс
                        if old_class_id in old_to_new_mapping:
                            new_class_id = old_to_new_mapping[old_class_id]
                            # Заменяем старый ID класса на новый
                            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                            filtered_lines.append(new_line)
                
                # Записываем отфильтрованные аннотации
                if filtered_lines:  # Записываем только если есть релевантные объекты
                    with open(output_label_path, 'w') as f:
                        f.writelines(filtered_lines)
                    
                    # Копируем соответствующий файл изображения
                    image_file = label_file.replace('.txt', '.jpg')  # Измените расширение при необходимости
                    image_path = os.path.join(images_dir, image_file)
                    output_image_path = os.path.join(out_images_dir, image_file)
                    
                    if os.path.exists(image_path):
                        shutil.copy2(image_path, output_image_path)
                else:
                    # Пропускаем копирование изображения, если нет релевантных объектов
                    print(f"Нет релевантных объектов в {label_file}, пропускаем изображение")

# Использование
if __name__ == "__main__":
    filter_annotations_for_2_classes('dataset', 'dataset_2classes')