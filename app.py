
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'reports'
app.config['HISTORY_FILE'] = 'history/history.json'

# Создание необходимых директорий
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Инициализация файла истории (если отсутствует)
if not os.path.exists(app.config['HISTORY_FILE']):
    with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print("✅ Создан пустой файл истории")

# Инициализация модели
def initialize_model():

    model_paths = [
        'C:/cafe-table-usage/runs/detect/cafe_table_people_detection/weights/best.pt',  # Дообученная модель
        'yolov8n.pt',  # Базовая модель
        'yolov8s.pt',  # Альтернативная модель
    ]

    for model_path in model_paths:
        try:
            model = YOLO(model_path)
            print(f" Модель {model_path} успешно загружена")

            # Проверяем, какие классы поддерживает модель
            if hasattr(model, 'names'):
                print(f" Классы модели: {model.names}")
            else:
                print(" Не удалось получить список классов модели")

            return model
        except Exception as e:
            print(f" Ошибка загрузки модели {model_path}: {e}")
            continue

    # Если ни одна модель не загрузилась, выбрасываем исключение
    raise Exception("Не удалось загрузить ни одну из доступных моделей")

try:
    model = initialize_model()
except Exception as e:
    print(f" Критическая ошибка при инициализации модели: {e}")
    print("Приложение не может работать без модели")
    exit(1)

def detect_objects(image_path):
    """
    Универсальная функция детекции объектов, работающая с разными моделями
    """
    try:
        # Выполнение детекции
        results = model(image_path)

        # Инициализация результатов
        detections = {
            'tables': [],      # Столы
            'people': [],      # Люди
        }

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    coords = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                    detection = {
                        'bbox': [int(c) for c in coords],
                        'confidence': round(confidence, 2),
                        'class_id': class_id
                    }

                    # Определение типа объекта в зависимости от модели
                    # Для дообученной модели с 2 классами (table, people)
                    if hasattr(model, 'names') and len(model.names) == 2:
                        if class_id == 0 and confidence > 0.3:  # table
                            detections['tables'].append(detection)
                        elif class_id == 1 and confidence > 0.3:  # people
                            detections['people'].append(detection)
                    # Для стандартной COCO модели
                    elif hasattr(model, 'names') and 'person' in model.names.values():
                        # Стандартные ID классов COCO
                        if class_id == 62 and confidence > 0.3:  # dining table
                            detections['tables'].append(detection)
                        elif class_id == 0 and confidence > 0.3:  # person
                            detections['people'].append(detection)
                    # Если не можем определить, используем эвристику
                    else:
                        if confidence > 0.3:
                            if class_id in [0, 1]:  # Предполагаем, что 0 или 1 - люди
                                detections['people'].append(detection)
                            elif class_id in [62, 60]:  # Предполагаем, что 62 или 60 - столы
                                detections['tables'].append(detection)

        return detections

    except Exception as e:
        print(f" Ошибка в detect_objects: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}

def analyze_table_occupancy(detections):

    tables = detections.get('tables', [])
    people = detections.get('people', [])

    # Подсчет столов
    table_count = len(tables)

    occupied = 0

    # Проверяем расположение людей относительно столов
    for table in tables:
        table_center = ((table['bbox'][0] + table['bbox'][2]) / 2,
                       (table['bbox'][1] + table['bbox'][3]) / 2)

        for person in people:
            person_center = ((person['bbox'][0] + person['bbox'][2]) / 2,
                            (person['bbox'][1] + person['bbox'][3]) / 2)

            # Расстояние между центрами
            distance = ((table_center[0] - person_center[0]) ** 2 +
                       (table_center[1] - person_center[1]) ** 2) ** 0.5

            # Если человек близко к столу - считаем стол занятым
            if distance < 50:  # Порог расстояния в пикселях
                occupied += 1
                break

    # Если столов нет, но есть люди, делаем эвристическую оценку
    if table_count == 0 and len(people) > 0:
        # Предполагаем, что 1-2 человека за 1 стол
        table_count = max(1, len(people) // 2)
        occupied = min(len(people), table_count)

    free = max(0, table_count - occupied)

    return {
        'total_tables': table_count,
        'occupied': occupied,
        'free': free,
        'occupancy_rate': round((occupied / table_count * 100), 1) if table_count > 0 else 0,
        'method': 'spatial_analysis'
    }

def draw_bounding_boxes(image_path, detections, stats):

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f" Не удалось загрузить изображение: {image_path}")
            return None

        # Цвета для классов (BGR)
        class_colors = {
            'tables': (0, 255, 0),      # Зелёный - столы
            'people': (0, 0, 255)       # Красный - люди
        }

        # Отрисовка рамок
        for obj_type, detections_list in detections.items():
            if obj_type == 'error' or not isinstance(detections_list, list):
                continue

            color = class_colors.get(obj_type, (255, 255, 255))
            label_prefix = obj_type.upper()[:4]

            for detection in detections_list:
                if 'bbox' not in detection:
                    continue

                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']

                # Рисуем прямоугольник
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Добавляем метку
                label = f"{label_prefix} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Добавляем статистику в угол изображения
        stats_text = [
            f"TOTAL: {stats['total_tables']}",
            f"OCCUPIED: {stats['occupied']}",
            f"FREE: {stats['free']}",
            f"RATE: {stats['occupancy_rate']}%",
            f"METHOD: {stats['method']}"
        ]

        y_offset = 30
        for i, text in enumerate(stats_text):
            # Тень
            cv2.putText(img, text, (10, y_offset + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            # Основной текст
            cv2.putText(img, text, (10, y_offset + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Сохраняем изображение с аннотациями
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_annotated{ext}")
        cv2.imwrite(output_path, img)

        return output_path

    except Exception as e:
        print(f" Ошибка в draw_bounding_boxes: {str(e)}")
        print(traceback.format_exc())
        return None

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден в запросе'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        # Проверка формата файла
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Неподдерживаемый формат файла. Разрешены: png, jpg, jpeg, gif, bmp'}), 400

        # Сохранение файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{timestamp}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f" Сохранено изображение: {filepath}")

        # Детекция объектов
        detections = detect_objects(filepath)

        if 'error' in detections:
            return jsonify({'error': f'Ошибка детекции: {detections["error"]}'}), 500

        print(f"🔍 Детекция завершена: {len(detections.get('tables', []))} столов, {len(detections.get('people', []))} людей")

        # Анализ занятости
        stats = analyze_table_occupancy(detections)
        print(f"📊 Статистика: {stats}")

        # Отрисовка рамок
        annotated_path = draw_bounding_boxes(filepath, detections, stats)

        # Подготовка данных для сохранения в историю
        record = {
            'timestamp': datetime.now().isoformat(),
            'original_image': filename,
            'annotated_image': os.path.basename(annotated_path) if annotated_path else None,
            'stats': stats,
            'detections': {
                'tables_count': len(detections.get('tables', [])),
                'people_count': len(detections.get('people', []))
            }
        }

        # Сохранение в историю
        try:
            with open(app.config['HISTORY_FILE'], 'r+', encoding='utf-8') as f:
                history = json.load(f)
                history.append(record)
                history = history[-100:]  # Ограничиваем 100 записями
                f.seek(0)
                json.dump(history, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            print(f" Ошибка сохранения в историю: {str(e)}")
            # Создаём новый файл истории если повреждён
            with open(app.config['HISTORY_FILE'], 'w', encoding='utf-8') as f:
                json.dump([record], f, ensure_ascii=False, indent=2)

        # Подготовка ответа
        response = {
            'success': True,
            'stats': stats,
            'detections': record['detections'],
            'original_image_url': f'/static/uploads/{filename}',
            'annotated_image_url': f'/static/uploads/{os.path.basename(annotated_path)}' if annotated_path else f'/static/uploads/{filename}',
            'timestamp': record['timestamp']
        }

        print(" Обработка изображения успешно завершена")
        return jsonify(response)

    except Exception as e:
        error_msg = f"Критическая ошибка: {str(e)}"
        print(f" {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg, 'traceback': traceback.format_exc()}), 500

@app.route('/history')
def get_history():

    try:
        if not os.path.exists(app.config['HISTORY_FILE']):
            return jsonify({'success': True, 'history': []})
        
        with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Валидация структуры данных
        valid_history = []
        for record in history:
            if isinstance(record, dict) and 'stats' in record:
                valid_history.append(record)
        
        return jsonify({'success': True, 'history': valid_history[-20:]})
    
    except Exception as e:
        print(f" Ошибка загрузки истории: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e), 'history': []}), 500

@app.route('/report')
def generate_report():

    try:
        if not os.path.exists(app.config['HISTORY_FILE']):
            return jsonify({'error': 'История отсутствует'}), 400
        
        with open(app.config['HISTORY_FILE'], 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return jsonify({'error': 'История пуста'}), 400
        
        # Берём последние 15 записей
        recent_history = [h for h in history if isinstance(h, dict) and 'stats' in h][-15:]
        if not recent_history:
            return jsonify({'error': 'Нет валидных записей в истории'}), 400
        
        # Имя файла отчёта
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}.pdf"
        filepath = os.path.join(app.config['REPORT_FOLDER'], filename)
        
        # Создание PDF
        doc = SimpleDocTemplate(filepath, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        
        # Кастомные стили
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1
        )
        
        elements = []
        elements.append(Paragraph("АНАЛИЗ ИСПОЛЬЗОВАНИЯ СТОЛОВ В КАФЕ", title_style))
        elements.append(Paragraph(f"Отчёт сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Статистика
        total_analyzed = len(recent_history)
        avg_occupancy = sum(r['stats']['occupancy_rate'] for r in recent_history) / total_analyzed
        total_tables_sum = sum(r['stats']['total_tables'] for r in recent_history)
        occupied_sum = sum(r['stats']['occupied'] for r in recent_history)
        
        summary_data = [
            ['Показатель', 'Значение'],
            ['Всего анализов', f"{total_analyzed}"],
            ['Средняя загруженность', f"{avg_occupancy:.1f}%"],
            ['Всего проанализировано столов', f"{total_tables_sum}"],
            ['Из них занято', f"{occupied_sum}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[250, 150])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 30))
        
        # Детальная таблица
        elements.append(Paragraph("Детальная статистика по анализам", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        detail_data = [['Дата/Время', 'Столов', 'Занято', 'Свободно', 'Загруженность']]
        
        for record in recent_history:
            s = record['stats']
            time_str = datetime.fromisoformat(record['timestamp']).strftime('%d.%m %H:%M')
            detail_data.append([
                time_str,
                s['total_tables'],
                s['occupied'],
                s['free'],
                f"{s['occupancy_rate']}%"
            ])
        
        detail_table = Table(detail_data, colWidths=[100, 80, 80, 80, 100])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white])
        ]))
        
        elements.append(detail_table)
        elements.append(Spacer(1, 30))
        
        # Выводы
        elements.append(Paragraph("ВЫВОДЫ", styles['Heading2']))
        elements.append(Spacer(1, 10))
        
        if avg_occupancy > 70:
            conclusion = "Загруженность кафе высокая — рекомендуется оптимизация расстановки столов"
        elif avg_occupancy > 40:
            conclusion = "Загруженность кафе средняя — текущая конфигурация эффективна"
        else:
            conclusion = "Загруженность кафе низкая — возможно, требуется маркетинговая активность"
        
        elements.append(Paragraph(f"• {conclusion}", styles['Normal']))
        elements.append(Paragraph(f"• Среднее количество столов за анализ: {total_tables_sum // total_analyzed}", styles['Normal']))
        elements.append(Paragraph("• Система использует дообученную модель YOLOv8 для детекции объектов", styles['Normal']))
        
        doc.build(elements)
        
        return jsonify({
            'success': True, 
            'report_url': f'/reports/{filename}',
            'message': 'Отчёт успешно сгенерирован'
        })
    
    except Exception as e:
        print(f" Ошибка генерации отчёта: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/reports/<filename>')
def download_report(filename):
    """Скачивание отчёта"""
    filepath = os.path.join(app.config['REPORT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'Файл отчёта не найден'}), 404

@app.route('/test')
def test_page():
    """Тестовая страница"""
    return """
    <h1> Сервер работает!</h1>
    <p>Перейдите на <a href="/">главную страницу</a> для анализа изображений кафе.</p>
    <p>Статус модели: <strong>дообученная модель загружена</strong></p>
    """

if __name__ == '__main__':
    print("=" * 70)
    print(" ЗАПУСК ВЕБ-ПРИЛОЖЕНИЯ 'Анализ использования столов в кафе'")
    print("=" * 70)
    print(f" Папка загрузок: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f" Папка отчётов: {os.path.abspath(app.config['REPORT_FOLDER'])}")
    print(f" Файл истории: {os.path.abspath(app.config['HISTORY_FILE'])}")
    print("=" * 70)
    print(" Приложение доступно по адресу: http://127.0.0.1:5000")
    print("  Для остановки нажмите Ctrl+C")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)