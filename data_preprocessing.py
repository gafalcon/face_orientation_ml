# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
up = pandas.read_csv("up.csv");

#TODO validate for missing values
def distance(row):
    if (row == 0).any():
        return -1
    return ((row[0]-row[2])**2 + (row[1]-row[3])**2)/(row[4])

def clean_data(filename):
    df = pandas.read_csv(filename+".csv")
    res = df.loc[:,("angle_leye_lear", "angle_reye_rear", "angle_leye_nose", "angle_reye_nose", "angle_lear_nose", "angle_rear_nose")]
    res["lear"] = df["l_ear_x"] != 0
    res["rear"] = df["r_ear_x"] != 0
    res["d_leye_lear"] = df.loc[:,("l_eye_x", "l_eye_y", "l_ear_x", "l_ear_y", "neck_ankle_height")].apply(distance, axis=1)
    res["d_leye_reye"] = df.loc[:,("l_eye_x", "l_eye_y", "r_eye_x", "r_eye_y", "neck_ankle_height")].apply(distance, axis=1)
    res["d_leye_nose"] = df.loc[:,("l_eye_x", "l_eye_y", "nose_x", "nose_y", "neck_ankle_height")].apply(distance, axis=1)
    res["d_reye_rear"] = df.loc[:,("r_eye_x", "r_eye_y", "r_ear_x", "r_ear_y", "neck_ankle_height")].apply(distance, axis=1)
    res["d_reye_nose"] = df.loc[:,("r_eye_x", "r_eye_y", "nose_x", "nose_y", "neck_ankle_height")].apply(distance, axis=1)
    res["d_nose_neck"] = df.loc[:,("nose_x", "nose_y", "neck_x", "neck_y", "neck_ankle_height")].apply(distance, axis=1)
    res["orientation"] = filename

    res.to_csv(filename+"_res.csv")
    
clean_data("up")
clean_data("down")
clean_data("left")
clean_data("right")
clean_data("center")