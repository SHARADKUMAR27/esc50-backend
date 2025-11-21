import pickle
import numpy as np

print("Loading label_encoder.pkl ...")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("Extracting class names...")

# label_encoder.classes_ contains the correct training labels
classes = le.classes_

print("Number of classes:", len(classes))
print("Example:", classes[:5])  # show first 5

np.save("esc50_classes.npy", classes)

print("esc50_classes.npy file created successfully!")
