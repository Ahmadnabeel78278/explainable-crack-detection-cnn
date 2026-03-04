import os
import re
import datetime
from flask import Flask, request, render_template, make_response
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, predict_and_explain
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import io
import random
import shutil
from datetime import datetime

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

@app.route('/download/pdf', methods=['POST'])
def download_pdf():
    prediction = request.form.get('prediction', 'N/A')
    confidence = request.form.get('confidence', 'N/A')
    explanation = request.form.get('explanation', '')
    image_file = request.form.get('image_file', '')
    heatmap_file = request.form.get('heatmap_file', '')

    # Clean explanation (remove HTML tags)
    explanation_plain = re.sub(r'<[^>]+>', '', explanation)
    explanation_plain = explanation_plain.replace('<br>', '\n')

    # Create PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 1 * inch
    line_height = 14
    y = height - margin

    # Title and timestamp
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Concrete Crack Detection Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # Images side by side at the top
    img_width = 3.5 * inch
    img_height = 3.5 * inch
    x_left = margin
    x_right = margin + img_width + 0.5*inch

    original_path = os.path.join('app', 'static', 'uploads', image_file)
    if os.path.exists(original_path):
        img = ImageReader(original_path)
        c.drawImage(img, x_left, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True)

    if heatmap_file and heatmap_file != 'None':
        heatmap_path_full = os.path.join('app', 'static', heatmap_file)
        if os.path.exists(heatmap_path_full):
            img = ImageReader(heatmap_path_full)
            c.drawImage(img, x_right, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True)

    y -= img_height + 30  # Move below images

    # Prediction and confidence
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Prediction: {prediction}")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Confidence: {confidence}")
    y -= 30

    # Explanation heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Explanation:")
    y -= 20

    # Function to wrap text
    def draw_wrapped_text(text, x, y, max_width, font_size, line_height):
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            if c.stringWidth(test_line, "Helvetica", font_size) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))

        c.setFont("Helvetica", font_size)
        for line in lines:
            if y < margin:  # Start new page if needed
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", font_size)
            c.drawString(x, y, line)
            y -= line_height
        return y

    # Draw wrapped explanation
    max_width = width - 2*margin
    y = draw_wrapped_text(explanation_plain, margin, y, max_width, 10, 14)

    c.save()
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers["Content-Disposition"] = "attachment; filename=crack_analysis.pdf"
    response.headers["Content-Type"] = "application/pdf"
    return response

@app.route('/download/txt', methods=['POST'])
def download_txt():
    prediction = request.form.get('prediction', 'N/A')
    confidence = request.form.get('confidence', 'N/A')
    explanation = request.form.get('explanation', '')
    image_file = request.form.get('image_file', '')
    heatmap_file = request.form.get('heatmap_file', '')

    # Clean explanation (remove HTML tags if any, but keep bullet points)
    explanation_plain = re.sub(r'<[^>]+>', '', explanation)  # strip HTML tags
    explanation_plain = explanation_plain.replace('<br>', '\n')  # replace <br> with newline

    content = f"""CONCRETE CRACK DETECTION REPORT

Prediction: {prediction}
Confidence: {confidence}

EXPLANATION:
{explanation_plain}

Images:
- Original: {image_file}
- Heatmap: {heatmap_file if heatmap_file != 'None' else 'Not generated'}

Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    response = make_response(content)
    response.headers["Content-Disposition"] = "attachment; filename=crack_analysis.txt"
    response.headers["Content-Type"] = "text/plain"
    return response

@app.route('/sample')
def sample():
    samples_dir = os.path.join('app', 'static', 'sample')
    if not os.path.exists(samples_dir):
        return "Sample folder not found", 404

    # Get list of image files in the samples directory
    sample_files = [f for f in os.listdir(samples_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not sample_files:
        return "No sample images available", 404

    # Pick a random sample
    chosen = random.choice(sample_files)
    sample_source = os.path.join(samples_dir, chosen)

    # Generate unique filename and copy to uploads
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'sample_{timestamp}_{chosen}'
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copy2(sample_source, dest_path)

    # Run prediction
    pred_class, confidence, explanation, heatmap_path = predict_and_explain(model, dest_path)
    explanation_html = re.sub(r'(?<!\A)\s*([-*] |\d+\. )', r'<br>\1', explanation)

    # Add to history
    history.append({
        'filename': filename,
        'prediction': pred_class,
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_path': 'uploads/' + filename,
        'heatmap_path': heatmap_path
    })
    if len(history) > 10:
        history.pop(0)

    return render_template('result.html',
                            image_file=filename,
                            heatmap_file=heatmap_path,
                            prediction=pred_class,
                            confidence=f"{confidence:.2%}",
                            explanation=explanation_html)
if __name__ == '__main__':
    app.run(debug=True)