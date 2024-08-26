import pandas as pd
from sklearn.linear_model import SGDClassifier

if __name__ == '__main__':    
    # Data Gathering
    df = pd.DataFrame({
        'movie':['Kalki','Tumbbad','Indiana Jones','Tiger 3'],
        'rating':[6.2,7.8,8.1,4.5],
        'released_date':[2024,2018,1990,2023],
        'watched':[1,1,0,0]
    })

    # Feature Engg
    df['recency'] = pd.Timestamp.today().year - df['released_date']

    # Training Data Prep
    X = df[['rating','recency']].to_numpy()
    y = df['watched'].to_numpy()

    # Model Training
    model = SGDClassifier(loss='log_loss',penalty=None,max_iter=100,learning_rate='constant',eta0=0.01,random_state=42)
    model.fit(X, y)
    print(model.coef_,model.intercept_)
    print(model.get_params())

    # Validating
    y_val_prob = model.predict_proba(X)
    y_val = model.predict(X)
    print([p2 for p1,p2 in y_val_prob], y_val)

    # Validation Accuracy
    val_acc = 100*sum(y_val==y)/len(y)
    print(f'validation accuracy = {val_acc}%')

