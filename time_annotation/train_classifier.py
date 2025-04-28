import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from annotation_utils import preprocess_image, extract_features, preprocess_ir_image, extract_ir_features, extract_rgb_highlevel_features, extract_ir_highlevel_features
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

root_folder = '../../data/yolo-labelled/test'
save_dir = './runs'

# time_of_day_to_label = {
#     "pre_sunrise": 0, "post_sunrise": 1, "noon": 2, "pre_sunset": 3, "post_sunset": 4,
#     "1.1": 0, "1.2": 1, "1.3": 2, "1.4": 3, "1.5": 4,
#     "2.1": 0, "2.2": 1, "2.3": 2, "2.4": 3, "2.5": 4,
# }

time_of_day_to_label = {
    "pre_sunrise": 0, "post_sunrise": 1, "noon": 2, "pre_sunset": 1, "post_sunset": 0,
    "1.1": 0, "1.2": 1, "1.3": 2, "1.4": 1, "1.5": 0,
    "2.1": 0, "2.2": 1, "2.3": 2, "2.4": 1, "2.5": 0,
}


train_dataset = []
test_dataset = []

for dirpath, dirnames, filenames in os.walk(root_folder):
    if 'ir' in dirpath.lower():
        continue
    for folder_name in time_of_day_to_label:
        if folder_name in dirpath:
            label = time_of_day_to_label[folder_name]
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dirpath, fname)
                    if folder_name.startswith("1."):
                        train_dataset.append((img_path, label))
                    elif folder_name.startswith("2."):
                        test_dataset.append((img_path, label))
            break

print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} testing image paths.")

def extract_features_from_dataset(dataset):
    X, y = [], []
    for img_path, label in dataset:
        try:
            img, hsv = preprocess_image(img_path)
            # img = preprocess_ir_image(img_path)
            features = extract_features(img, hsv)
            # features = extract_ir_features(img)
            # features = extract_rgb_highlevel_features(img, hsv)
            # features = extract_ir_highlevel_features(img)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"[SKIPPED] {img_path} — {e}")
    return np.array(X), np.array(y)

X_train, y_train = extract_features_from_dataset(train_dataset)
X_test, y_test = extract_features_from_dataset(test_dataset)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    # class_weight='balanced',
    random_state=45,
    max_features='sqrt'
)

# clf = XGBClassifier(
#     objective='multi:softmax',
#     num_class=5,
#     eval_metric='mlogloss',
#     max_depth=6,
#     n_estimators=300,
# )

clf.fit(X_train, y_train)

model_output_path = os.path.join(save_dir, 'time_of_day_classifier.tz')
joblib.dump(clf, model_output_path, compress=3)
print(f"Saved model to {model_output_path}")

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

for true, pred in zip(y_test[:20], y_pred[:20]):
    print(f"True: {true}, Predicted: {pred}")

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title("Confusion Matrix")
# plt.show()