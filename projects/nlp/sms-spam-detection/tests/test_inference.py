from src. inference import load_models, predict_messages


def test_load_model():
    transformer, vectorizer = load_models()
    
    assert transformer is not None
    assert vectorizer is not None
    
    


def test_predict_messages():
    transformer, vectorizer = load_models()
    test_message = [
        "Hello mom, where are you please call me we you are at home",
        "Congradulations, you have just won the biggest prise ever"
    ]
    
    predictions, probabilities = predict_messages(
        test_message, 
        transformer,
        vectorizer)


    assert len(predictions) == len(test_message)
    
    assert len(probabilities) == len(test_message)
    
    assert all(label in {"ham", "spam"} for label in predictions)
    
    assert all(0 <= p  <=1 for p in probabilities)
    
    assert predictions[0] == "ham"
    assert predictions[1] == "spam"    
    
