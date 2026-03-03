import os
import re
import datetime
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, predict_and_explain

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model()

# In-memory history (last 10 entries)
history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global history
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            pred_class, confidence, explanation, heatmap_path = predict_and_explain(model, filepath)
            # Format explanation: replace each bullet (starting with "- ") with a line break + bullet
            explanation_html = re.sub(r'(?<!\A)\s*([-*] |\d+\. )', r'<br>\1', explanation)

            # Add to history
            history.append({
                'filename': filename,
                'prediction': pred_class,
                'confidence': confidence,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': 'uploads/' + filename,
                'heatmap_path': heatmap_path
            })
            # Keep only last 10
            if len(history) > 10:
                history.pop(0)

            return render_template('result.html',
                                    image_file=filename,
                                    heatmap_file=heatmap_path,
                                    prediction=pred_class,
                                    confidence=f"{confidence:.2%}",
                                    explanation=explanation_html)
    return render_template('index.html', history=history[::-1])  # show newest first

if __name__ == '__main__':
    app.run(debug=True)