from ranked import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="Path to hdf5 database")
ap.add_argument("-m", "--model", required=True, help="Path to pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

print("[INFO] rank-1 accuracy : {:.2f}".format(rank1 * 100))
print("[INFO] rank-5 accuracy : {:.2f}".format(rank5 * 100))

db.close()







