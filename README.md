# API для распознования рукописного русского текста и типа документа.
Наше API получает на вход фотографию и возвращает json файл в котором находится текст в следующем формате: текст, его координаты, подпись/не подпись и тип файла.
## Содержание
* [Установка](#установка)    
* [Разметка данных](#разметка-данных)  
* [Преобразование форматов](#преобразование-форматов)  
* [Обучение модели YOLOv5](#обучение-модели-yolo11)  
* [Обучение нейросети CNN](#обучение-нейросети-cnn)  
* [Применение трансформеров](#применение-трансформеров)    
* [Определение типа документа](#определение-типа-документа)
* [Создание API](#создание-api)
* [Метрики](#метрики)
* [Пример](#пример)
  
## Установка
* *Установите  необходимые  библиотеки:*  `pip install -r requirements.txt`
## Разметка данных
* Сначала мы разметили фотографии, создавая аннотации, которые позволили выделить ключевые области текста.
## Преобразование форматов
* Разметку в формате JSON преобразовали в YAML для удобства дальнейшей работы.
## Обучение модели YOLO11
* Использовали модель YOLO11 для обнаружения текста на изображениях, что стало основой для дальнейшей обработки.  
## Обучение нейросети CNN
* На основе обработанных данных была обучена сверточная нейронная сеть (CNN RESNET) для распознавания рукописного текста.  
## Применение трансформеров
* Для повышения точности распознавания текста использовали архитектуру трансформеров.  
## Определение типа документа
* Добавили функциональность для определения типа файла на основании изображений документов.
## Создание API
* Разработка операций CRUD была успешно реализована с интеграцией базы данных, а также созданием RESTful API для удобного взаимодействия.
## Метрики
Выбранные метрики:
* CER
* WER
* Accuracy
## Пример
* Входное изображение:
<img src="photos/photo1" width="300" height="300"/>
* Выходной файл:
<img src="photos/photo2" width="300" height="300"/>

## Инструкция по использованию 
* Скачать файл main.ipynb для ввода пути файла с клавиатуры или foreapi.py для взятия из папки /temp/image_name.txt
* Скачать файлы model.pt и best.pt
* Запустить main введя путь в консоль или в файлик соответственно
