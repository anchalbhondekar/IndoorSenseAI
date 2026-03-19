from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.predict import evaluate

def main():

    print("🚀 Loading data...")
    train, test = load_data()

    print("🧹 Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(train, test)

    print("🤖 Training model...")
    model = train_model(X_train, y_train)

    print("📊 Evaluating...")
    accuracy = evaluate(model, X_test, y_test)

    print(f"✅ Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()