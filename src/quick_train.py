from ultralytics import YOLO

def quick_train():
    """Quick training script optimized for hackathon timeline"""
    
    # Load pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Quick training with minimal epochs for demo
    results = model.train(
        data='data/dataset/data.yaml',
        epochs=10,              # Very quick training - reduced to 10 epochs
        imgsz=640,
        batch=16,
        name='quick_kidney_stone',
        project='models',
        lr0=0.001,              # Lower LR for transfer learning
        patience=10,            # Early stopping
        save_period=5,
        cache=False,
        workers=4,
        amp=True
    )
    
    # Save model
    model.save('models/quick_kidney_stone.pt')
    
    print("Quick training completed!")
    print("Model saved as: models/quick_kidney_stone.pt")
    
    return results

if __name__ == "__main__":
    quick_train()
