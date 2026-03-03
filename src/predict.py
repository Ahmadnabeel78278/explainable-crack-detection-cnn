import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.cm as cm
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# --- Helper to discover available models (silent) ---
def get_working_model():
    """Return the name of a model that supports generateContent without printing."""
    try:
        models = client.models.list()
        # No printing of the full list
        for m in models:
            if 'generateContent' in m.supported_actions:
                # Prefer gemini-2.5-flash, then gemini-2.0-flash, then gemini-1.5-flash
                if 'gemini-2.5-flash' in m.name:
                    return m.name
                elif 'gemini-2.0-flash' in m.name:
                    return m.name
                elif 'gemini-1.5-flash' in m.name:
                    return m.name
        # Fallback: first model that supports generateContent
        for m in models:
            if 'generateContent' in m.supported_actions:
                return m.name
    except Exception as e:
        print(f"Error listing models: {e}")  # Keep error message
        return "models/gemini-2.5-flash"  # fallback
    return None

WORKING_MODEL = get_working_model()
print(f"Using model: {WORKING_MODEL}")  

def extract_full_text(response):
    """
    Safely extract full text from Gemini response.
    Handles both response.text and candidates structure.
    """
    try:
        # First try normal shortcut
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # Fallback: manually collect from candidates
        full_text = ""
        if hasattr(response, "candidates"):
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            full_text += part.text

        return full_text.strip()

    except Exception as e:
        return f"Explanation extraction error: {str(e)}"

# Custom functions for model loading
def _mean_keepdims(x):
    import tensorflow as tf
    return tf.reduce_mean(x, axis=-1, keepdims=True)

def _max_keepdims(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

def load_model(model_path='models/crack_detection_mobilenet_cbam.h5'):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            '_mean_keepdims': _mean_keepdims,
            '_max_keepdims': _max_keepdims
        }
    )

def predict_image(model, image_path, img_size=(224,224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    pred = model.predict(img_array)[0][0]
    return pred

def explain_with_gemini(pred_class, confidence):

    prompt = f"""
Prediction:
Class: {pred_class}
Confidence: {confidence:.2f}

Provide a clear explanation in 4 bullet points using '-' only.
Keep it under 180 words.
Be specific to this prediction.
Avoid markdown formatting.
"""

    try:
        response = client.models.generate_content(
            model=WORKING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=800,   # Increased to prevent cut-off
            )
        )
        return extract_full_text(response)

    except Exception as e:
        return f"Explanation unavailable: {str(e)}"


def generate_gradcam(model, img_array, target_layer_name='multiply_1'):
    """
    Generate Grad-CAM heatmap for the predicted class.
    Handles cases where model output might be a list.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        # outputs is a list: [conv_output, predictions]
        conv_output = outputs[0]
        predictions = outputs[1]
        
        # If predictions is a list (e.g., for multiple outputs), take the first element
        if isinstance(predictions, list):
            pred_tensor = predictions[0]
        else:
            pred_tensor = predictions
        
        # For binary classification, loss is the score for the predicted class
        loss = pred_tensor[:, 0]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    
    conv_output = conv_output[0]
    pooled_grads = pooled_grads[0]
    
    weighted_conv = conv_output * pooled_grads
    heatmap = tf.reduce_sum(weighted_conv, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    
    return heatmap.numpy()

def overlay_heatmap(original_img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    overlayed = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
    overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlayed_bgr)

def predict_and_explain(model, image_path):
    pred = predict_image(model, image_path)
    pred_class = 'Crack' if pred > 0.5 else 'No Crack'
    confidence = pred if pred > 0.5 else 1 - pred
    explanation = explain_with_gemini(pred_class, confidence)
    
    heatmap_path = None
    if pred_class == 'Crack':
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)
        
        heatmap = generate_gradcam(model, img_array)
        
        heatmap_filename = 'heatmap_' + os.path.basename(image_path)
        static_uploads = os.path.join('app', 'static', 'uploads')
        os.makedirs(static_uploads, exist_ok=True)
        heatmap_filepath = os.path.join(static_uploads, heatmap_filename)
        overlay_heatmap(image_path, heatmap, heatmap_filepath)
        heatmap_path = 'uploads/' + heatmap_filename
    
    return pred_class, confidence, explanation, heatmap_path