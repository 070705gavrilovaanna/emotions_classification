from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.nn import Embedding, LSTM, Module, Dropout, Linear, ReLU, BatchNorm1d

class LSTMModel(Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_classes=6):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.batchnorm = BatchNorm1d(hidden_size * 2)
        self.dropout = Dropout(0.3)
        self.fc1 = Linear(hidden_size * 2, hidden_size)
        self.fc2 = Linear(hidden_size, num_classes)
        self.relu = ReLU()

    def forward(self, x):
        x = self.embedding(x)
        output, (_, _) = self.lstm(x)
        x = torch.mean(output, dim=1)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(le.classes_)

model = LSTMModel(vocab_size, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_LSTM.pth', map_location=device))
model.eval()

emotions = le.classes_.tolist()

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return emotions[pred]

app = FastAPI()

html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Classifier</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }}
        textarea {{
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            resize: vertical;
            font-family: inherit;
            transition: border-color 0.3s;
        }}
        textarea:focus {{
            outline: none;
            border-color: #667eea;
        }}
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 25px;
            cursor: pointer;
            margin-top: 15px;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }}
        .result {{
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        .result-text {{
            font-size: 18px;
            color: #333;
            margin-bottom: 10px;
            word-wrap: break-word;
        }}
        .result-emotion {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
            margin-top: 10px;
        }}
        hr {{
            margin: 20px 0;
            border: none;
            height: 1px;
            background: #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Определение эмоций в тексте</h1>
        <form method="post">
            <textarea name="text" rows="5" placeholder="Введите текст на английском..."></textarea>
            <button type="submit">Определить эмоцию</button>
        </form>
        {result}
    </div>
</body>
</html>
"""

@app.get("/")
def home():
    return HTMLResponse(html_form.format(result=""))

@app.post("/")
def predict(text: str = Form(...)):
    emotion = predict_emotion(text)
    result_html = f"""
    <div class="result">
        <div class="result-text"><strong>Ваш текст:</strong> {text}</div>
        <hr>
        <div class="result-emotion">Эмоция: {emotion}</div>
    </div>
    """
    return HTMLResponse(html_form.format(result=result_html))