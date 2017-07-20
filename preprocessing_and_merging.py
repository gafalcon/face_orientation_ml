import pandas
from os import listdir
from os.path import isfile, join
import sys

if len(sys.argv) < 3:
    print "Ingrese ubicacion de archivos csv y nombre de csv resultado"
    sys.exit(-1)

filepath = sys.argv[1]
print "Merging files in", filepath
result_csv_file = sys.argv[2]

def clean_data(filename):
    print filename,
    if filename.startswith("up"):
        orientation = "up"
    elif filename.startswith("down"):
        orientation = "down"
    elif filename.startswith("left"):
        orientation = "left"
    elif filename.startswith("right"):
        orientation = "right"
    elif filename.startswith("center"):
        orientation = "center"
    elif filename.startswith("tv"):
        orientation = "tv"
    else:
        return
    res = pandas.read_csv(join(filepath, filename))
    res = res[res.columns.drop(['frame', 'person'])]
    res["orientation"] = orientation
    # res.to_csv(filename+"_res.csv", index=False)
    return res

csvfiles = [clean_data(f) for f in listdir(filepath) if f.endswith(".csv")]
training_df = pandas.concat(csvfiles)

training_df.to_csv(join("datasets","training_datasets",result_csv_file), index=False)
print "\nCreated training dataset ",result_csv_file
