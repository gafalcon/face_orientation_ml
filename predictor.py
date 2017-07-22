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
14. lear: 1 if present 0 otherwise
15. rear: 1 if present 0 otherwise

"""
classifier = None#joblib.load("D:/RAP_Openpose/tree_model_full.pkl")
scaler = None#joblib.load("D:/RAP_OPenpose/tree_scaler_full.pkl")
class_names = ['center', 'down', 'left', 'right', 'tv', 'up']

def load_predictor(scaler_pkl,classifier_pkl):
    global classifier, scaler
    print "loading classifier ",classifier_pkl,"...",
    classifier = joblib.load(classifier_pkl)
    print "Classifier loaded"
    print "loading scaler ",scaler_pkl,"...",
    scaler = joblib.load(scaler_pkl)
    print "Scaler loaded"
    return True

def predict(*features):
    features_to_scale = np.array(features[:-2], dtype=np.float64)
    try:
        scaled = scaler.transform(features_to_scale.reshape(1,-1))
        scaled_features = np.append(scaled,features[-2:])
        pred = classifier.predict(scaled_features.reshape(1,-1))
        return class_names[pred[0]]
    except ValueError as e:
        print features
        return "none"

