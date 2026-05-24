# Классификация эмоций в тексте: LSTM vs Transformer (с API для инференса)

Проект по классификации эмоций в текстовых данных (anger, fear, joy, love, sadness, surprise). Сравниваются две архитектуры: LSTM с Bidirectional слоями и кастомный Transformer с позиционным кодированием. Обе модели обучались с нуля на датасете emotions. Лучшая модель (LSTM) развёрнута в виде REST API на Hugging Face Spaces.

**Ссылка на проект:** [emotions_classification.ipynb](https://colab.research.google.com/drive/1HVAgkQcAQ6cQES5qUjd9fux4qF9rq38R#scrollTo=4io7W_OW6lUR)

**Источник данных:** [Emotions Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=val.txt) (Kaggle)

**API на Hugging Face:** [emotion-classifier-api](https://huggingface.co/spaces/user43242422/emotion-classifier-api)


---

## Этапы работы

**Технологии:** Python, PyTorch, Scikit-learn, TensorFlow (Tokenizer), NumPy, Matplotlib

### 1. Загрузка и подготовка данных
- Распаковка датасета из train.txt, val.txt, test.txt
- Формат данных: `текст;эмоция` (6 классов: anger, fear, joy, love, sadness, surprise)
- Размер выборки: train - 16000, val - 2000, test - 2000
- Стратифицированное разделение для сохранения баланса классов

### 2. Токенизация и предобработка
- Токенизация через Keras Tokenizer (20000 слов, oov_token='<unk>')
- Паддинг до 100 токенов (padding='post', truncating='post')
- Кодирование меток через LabelEncoder
- Создание DataLoader'ов с batch_size=16

### 3. Моделирование

Сравнила 2 архитектуры:

**LSTM:**
- 2 слоя Bidirectional LSTM (hidden_size=128, embed_size=128)
- BatchNorm1d
- Dropout 0.3
- 2 полносвязных слоя с ReLU

**Transformer:**
- PositionalEncoding
- 2 слоя TransformerEncoder (nhead=4)
- Max pooling вместо Mean pooling
- Dropout 0.5

### 4. Обучение
- 20 эпох для каждой модели
- Optimizer: Adam (weight_decay=1e-4, lr=0.001)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Функция потерь: CrossEntropyLoss
- Сохранение лучших весов по F1-score на валидации

---

## Результаты

### LSTM (лучшая модель)
**Test F1 (macro): 0.95 | Test Accuracy: 0.95**

| Класс | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| anger | 0.95 | 0.96 | 0.96 |
| fear | 0.94 | 0.91 | 0.93 |
| joy | 0.98 | 0.93 | 0.95 |
| love | 0.93 | 0.98 | 0.96 |
| sadness | 0.99 | 0.95 | 0.97 |
| surprise | 0.92 | 0.99 | 0.95 |

### Transformer
**Test F1 (macro): 0.87 | Test Accuracy: 0.87**

| Класс | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| anger | 0.89 | 0.94 | 0.91 |
| fear | 0.98 | 0.61 | 0.75 |
| joy | 0.97 | 0.83 | 0.89 |
| love | 0.84 | 0.96 | 0.90 |
| sadness | 0.85 | 0.91 | 0.88 |
| surprise | 0.77 | 0.98 | 0.86 |

---

## API на Hugging Face

Обученная LSTM-модель доступна в виде REST API через Hugging Face Spaces.

**Технологии деплоя:** FastAPI, Docker, Hugging Face Spaces

**Файлы в репозитории:**
- `app.py` - основной код API (загрузка модели, обработка запросов, HTML-интерфейс встроен в код)
- `best_LSTM.pth` - веса обученной модели (13.5 MB)
- `tokenizer.pickle` - токенизатор для преобразования текста
- `label_encoder.pickle` - декодер меток эмоций
- `requirements.txt` - зависимости проекта
- `Dockerfile` - конфигурация для сборки контейнера

---

## Итоги

LSTM превосходит Transformer: Test F1 0.95 против 0.87. LSTM стабильно обучается (val_loss: 0.319 → 0.220), все классы выше F1=0.93. Transformer нестабилен (скачки val_loss) и проваливается на классе fear (recall 0.61). Причина: для Transformer с нуля нужно значительно больше данных. LSTM весит 13.5 MB и подходит для деплоя на CPU с ответом менее 100 мс.

---

## Возможные улучшения

1. **Weighted loss для редких классов** — если в новых данных будет дисбаланс, обычная loss проигнорирует редкие классы. Weighted loss увеличивает штраф за ошибки на них.
2. **Аугментация текста (back-translation, синонимы)** — создаёт новые варианты текстов без сбора данных. Модель перестаёт запоминать конкретные слова и учится понимать смысл.
3. **Предобученный BERT** — в отличие от моего Transformer с нуля, BERT уже обучен на миллионах документов. Дообучение на эмоциях даст качество выше LSTM, но модель станет тяжелее (400-500 MB) и медленнее.
