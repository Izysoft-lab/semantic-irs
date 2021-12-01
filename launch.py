from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

import numpy as np
import pandas as pd


app = Flask(__name__)
CORS(app)

@app.route('/')
def welcome():
    return render_template('index.html')
    
@app.route('/decision', methods=['POST'])
def predict():
    if request.method == 'POST':
        val = 50
        data = request.get_json()
        #data = {'thal':1,'ca':0, 'slope':0, 'oldpeak':2.5, 'exang':0, 'thalach':150, 'restecg':0,'fbs':1, 'chol':230, 'trestbps':140, 'cp':2, 'sex':1, 'age':60}
        print(data)
        ar = np.array([[data['age'], data['sexe'], data['niveau'], data['pression'], data['cholesterol'], 
        data['glycemie'], data['electrocardio'], data['frequence'], data['angine'], data['decalage'], data['pente'], data['fluoroscopie'], data['thalassemie']]])
        df2 = pd.DataFrame(ar, columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

        val = my_pipeline.predict_proba(df2.loc[0:0])[0][1] * 100
        print(val)
        return jsonify({"resultat": val})


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    # We are reading our data
    df = pd.read_csv("./heart.csv")

    y = df.target.values
    X = df.drop(['target'], axis = 1)

    # Normalize
    x = (X - np.min(X)) / (np.max(X) - np.min(X)).values

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    categorical_cols = ['sex', 'cp', 'thal', 'slope', 'exang', 'restecg']
    numerical_cols = [col for col in df.columns if (col not in categorical_cols and col != 'target')]

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=1000, random_state=1)

    from sklearn.metrics import mean_absolute_error

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)
                                ])

    my_pipeline.fit(x_train, y_train)
    app.run(port=5002)