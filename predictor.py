from sklearn.externals import joblib
import numpy as np

"""
Reads argumnents to predict the face orientation
Arg_order:
1. angle_leye_lear
2. angle_reye_rear
3. angle_leye_nose
4. angle_reye_nose
5. angle_lear_nose
6. angle_rear_nose
7. d_leye_lear
8. d_leye_reye
9. d_leye_nose
10. d_reye_rear
11. d_reye_nose
12. d_nose_neck
13. neck_x
12. lear: 1 if present 0 otherwise
13. rear: 1 if present 0 otherwise

"""
classifier = joblib.load("./tree_model_full.pkl")
scaler = joblib.load("./tree_scaler_full.pkl")
class_names = ['center', 'down', 'left', 'right', 'tv', 'up']

def predict(*features):
    features_to_scale = np.array(features[:-2], dtype=np.float64)
    scaled = scaler.transform(features_to_scale.reshape(1,-1))
    scaled_features = np.append(scaled,features[-2:])
    pred = classifier.predict(scaled_features.reshape(1,-1))
    return class_names[pred[0]]

