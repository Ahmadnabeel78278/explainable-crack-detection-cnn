from src.predict import load_model, predict_and_explain
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_prediction.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    
    model = load_model()
    pred_class, confidence, explanation = predict_and_explain(model, image_path)
    
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nExplanation:")
    print(explanation)