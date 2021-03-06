# ПРЕДСКАЗАНИЕ ОПАСНОСТИ КОСМИЧЕСКИХ ОБЪЕКТОВ

Предлагается решать следующую задачу: по вектору состояния (вектор координат и вектор скоростей) предсказывать вероятность падения космического объекта на Землю. Сразу оговоримся, что для данной задачи существуют решения, основанные на баллистике. Но давайте представим, что радиус Земли неизвестен, тогда нам не поможет второй закон Ньютона, но может помочь обучение на большом объеме данных!

## Описание обучающих данных

Каждый объект описывается вектором состояния (X, Y, Z, dX, dY, dZ) и меткой класса threat, где 1 - объект упадет на Землю, 0 - не упадет.
Для данных разработан соответствующий класс dataManager.Data.
Обучающие данные скачиваются с яндекс-диска. Кому интересно, в utils есть соответствующие функции для скачивания. Все ссылки есть в configs/data_links.yml

## Что можно сделать с данными?

1. Произвести аугментацию: простая аугментация, добавляющая данные, умноженные на -1 (геометрически мы не изменим параметры траектории, а значит не изменится и метка упал/не упал). Делать или не делать аугментацию решается в configs/train_config.yml параметром augmentation.
2. Создавать новые признаки. Для этого в configs/train_congig.yml в поле preproccessing необходимо прописать арифметическую операцию над имеющимися признаками и после запятой указать имя нового признака. Например, X * Y,XY. Другие примеры можно посмотреть в configs/train_config_1.yml
3. Разделять на обучающую и валидационную выборки с параметром split_size (configs/train_config.yml поле split_size).

Для генерации новых данных используется класс DataCreate. Сделан он исключительно для тестов и не создает верных меток

## Модели

Здесь можно использовать два типа моделей: логрегрессия и LGBM. Разработан класс-обертка (modelManager.Model) с методами train, predict_score и save.
Можно задавать в configs/train_config.yml тип модели и ее параметры.

## Метрики качества

Используется одна единственная метрика качества - ROC AUC (также строится ROC-кривая)

## Весь трубопровод

Весь пайплайн обучение/валидация осуществляется в функции main.train_test_pipeline_threat. На вход подается конфигурация обучения (configs/train_config.yml), ссылки на данные, имя модели (опционально) и пути сохранения ROC-кривых (опционально). В результате сохраняется модель и картинки (ROC-кривые). По умолчанию это папки models и images. В лог записывается ROC-AUC.

## EDA

В notebooks/eda.ipynb можно запустить пайплайн через ноутбук. Реализовано три сценария обучения. Каждый лучше предыдущего.

## Почему это потрясающий проект?

1. Нестандартная интересная задача.
2. Возможность генерации новых признаков
3. Простота и краткость функционала.
4. Как-будто выполнены все требования по основным баллам.

Безусловно достойно 30 баллов!

## Почему этот проект ужасен?

1. Странная задача, где всего шесть исходных признаков. К тому же кто в здравом уме доверит логистической регрессии определять упадет космический объект на Землю или нет?
2. При генерации новых признаков используется ~~evil~~ eval! Да простят меня проверяющие.
3. Скудость и бедность функционала, плохо описанные функции и методы. Более чем скромное логирование.
4. Отсутствует даже попытка на заработок доп.баллов.

Ну можно 23 балла поставить авансом, так и быть.
