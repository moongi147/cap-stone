from sklearn.metrics import confusion_matrix
from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
from matplotlib import cm
import matplotlib.pyplot as plt
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

model = load_model('./model/capstone_classification_real.model')

caltech_dir = "test"

X = []
filenames = []
files = glob.glob(caltech_dir + "/*.*")
image_w = 64
image_h = 64

pixels = image_h * image_w * 3

for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)

prediction = model.predict(X)

y_true=[]
row_num = 900

for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    pre_ans_str = ''
    y_true.append(pre_ans)

print(y_true)