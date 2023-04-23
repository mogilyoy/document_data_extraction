# Решение

### Общий путь

1. Предобработка данных *train.json*, создание новых выборок
2. Реализация базовой модели 
3. Попытка реализации classy_classification
4. Обучение предобученной SetFit модели
5. Применение Setfit модели на неразмеченных данных
6. Применение базовой модели к неразмеченным данным
7. Улучшение разбиения текстов на фрагменты, улучшение accuracy на трейне


### Вывод 
По ходу исследования была реализована модель градиентного бустинга для разметки фрагментов текста. Максимальный результат accuracy на трейне показала пара моделей XGBClassifier из библиотеки xgboost - 70.3% попаданий.

Улучшить полученный результат можно: 

*   Добавлением новых фич в обучающую выборку
*   Проработкой обучающей выборки для моделей
*   Использовать более сложные модели, нейронные сети
*   Улучшением качества разбиения текстов
*   Использовать готовые модели nlp для решения задачи NER, text summarization, information extraction (spacy и др.)

### XGBClassifier 
Эту модель я изначально решил использовать в качестве базовой модели для дальнейшего сравнения моделей. После обучения SetFit модели, реализовал несколько аналогичных моделей, но попытки улучшить результат, ухудшали метрику accuracy. Лучшая пара моделей классификации текстов показала accurcy хуже - 0.66 на трейне. По итогу была дотюнена изначальная пара моделей.


### SetFit 
Set Fit - обеспечивает высокую точность при небольшом объёме данных. В нашем случае в искомых предложениях часто используются одни и те же слова, поэтому объем уникальных элементов в обучающей выборке очень небольшой. К сожалению, не удалось добиться хороших результатов при помощи модели, так как на массивах больше 30 элементов, модель обучается очень долго (6-8 часов на 100 записях на google colab). Также безусловно повлияло то, что в качестве обучающей и валидационной выборки не были использованы леммы слов предложений, а модель обучалась на "сырых" фрагментах разбиения. При использовании нормальных форм слов, вероятно, можно добиться больших результатов. Также стоит отметить, что полученная модель очень тяжеловесна, предсказание на трейне размером 1800 текстов занимает 12 часов (на google colab), так как для предсказания каждого фрагмента требуется приблизительно 15 секунд. 

### Classy clssification
Classy clssification - удобный интерфейс для задачи текстовой классификации в библиотеке spacy. К сожалению, русскоязычные модели работают неправильно, поэтому от реализации пришлось отказаться.

### Список литературы
- [How to Summarize Text Using Python and Machine Learning](https://www.youtube.com/watch?v=SNimr_nOC7w)
- [Text summarization using spaCy](https://medium.com/analytics-vidhya/text-summarization-using-spacy-ca4867c6b744)
- [Постановка задачи автоматического реферирования и методы без учителя](https://habr.com/ru/articles/595517/)
- [GazetaSummarization](https://colab.research.google.com/drive/1B26oDFEKSNCcI0BPkGXgxi13pbadriyN)
- [classy-classification GitHub repo](https://github.com/Pandora-Intelligence/classy-classification)
- [Text Classification methods in NLP with deep learning](https://github.com/brightmart/text_classification)
- [huggingface transformers.pipeline()](https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html)
- [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)
- [Введение в библиотеку Transformers и платформу Hugging Face](https://habr.com/ru/articles/704592/)
- [Пайплайн для создания классификации текстовой информации](https://telegra.ph/Pajplajn-dlya-sozdaniya-klassifikacii-tekstovoj-informacii-04-13)
- [Мультиклассовая классификация текста.](https://habr.com/ru/articles/677512/)






