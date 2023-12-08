import cv2
import pygame
import Image, ImageTk
from tkinter.filedialog import *
from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
import time

win = Tk()
win.geometry("1024x768+500+20")
win.title("인공지능을 활용한 회전체 고장 진단")
win.option_add("*Font", "맑은고딕 25")
win.resizable(False, False) # 창 크기 변경 불가
win['bg'] = 'alice blue'

orb = Tk()
orb.geometry("490x490+10+320")
orb.title("실시간 오르빗 그래프와 스펙트럼")
orb.option_add("*Font", "맑은고딕 20")
orb.resizable(False, False)
orb['bg'] = 'alice blue'

label10 = Label(orb, text="실시간 오르빗 그래프와 스펙트럼")
label10.place(x=40, y=20)
label10['bg'] = 'alice blue'

label1 = Label(win, text="인공지능을 활용한 고장 진단")
label1.place(x = 300, y = 20)
label1['bg'] = 'alice blue'

photo = PhotoImage(file="C:\\Users\\id030\\PycharmProjects\\my-PT\\pythonProject8\\photo.gif")
label3 = Label(win, image=photo, width=969, height = 513)
label3.place(x=60, y=90)

def clear_photo_image():
    label3.config(image = '')
    label3['bg'] = 'alice blue'

def show_label3():
    label3.config(image = photo)

caltech_dir = "test"
image_w = 64
image_h = 64

pixels = image_h * image_w * 3

def diagnosis():  # 여기 추가할거면 들여쓰기 해야함
    pygame.init()  # <-- initializes video
    pygame.mixer.init()
    siren_sound = pygame.mixer.Sound("C:\\Users\\id030\\PycharmProjects\\my-PT\\pythonProject8\\warn.mp3")


    X = []
    filenames = []
    files = glob.glob(caltech_dir + "/*.*")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        filenames.append(f)
        X.append(data)

    X = np.array(X)

    model = load_model('./model/capstone_classification_real.model')
    # model = load_model('./model/capstone_classification_by_resnet_real.model')

    prediction = model.predict(X)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cnt = 0

    for i in prediction:
        pre_ans = i.argmax()  # 예측 레이블
        pre_ans_str = ''

        if pre_ans == 0:
            pre_ans_str = "미스얼라이먼트"
        elif pre_ans == 1:
            pre_ans_str = "정상"
        elif pre_ans == 2:
            pre_ans_str = "언밸런스"
        else:
            pre_ans_str = "상태 파악 불가능"
        if i[0] == 1:
            clear_photo_image()
            label2 = Label(win, text="현재 상태는 " + pre_ans_str + "로 추정됩니다.", font=("맑은고딕", 23), fg="red", bg="alice blue")
            label2.place(x=400, y=700)
            IMG = cv2.imread(files[cnt])
            dst = cv2.resize(IMG, dsize=(0, 0), fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow('diagnosis')
            cv2.imshow('diagnosis', dst)
            cv2.moveWindow('diagnosis', 690, 120)
            siren_sound.play(0)
            label2.after(7000, label2.destroy)

        elif i[1] == 1:
            clear_photo_image()
            label2 = Label(win, text="현재 상태는 " + pre_ans_str + "으로 추정됩니다.", font=("맑은고딕", 23), fg="green", bg="alice blue")
            label2.place(x=400, y=700)
            IMG = cv2.imread(files[cnt])
            dst = cv2.resize(IMG, dsize=(0, 0), fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow('diagnosis')
            cv2.imshow('diagnosis', dst)
            cv2.moveWindow('diagnosis', 690, 120)
            label2.after(7000, label2.destroy)


        elif i[2] == 1:
            clear_photo_image()
            label2 = Label(win, text="현재 상태는 " + pre_ans_str + "로 추정됩니다.", font=("맑은고딕", 23), fg="red", bg="alice blue")
            label2.place(x=400, y=700)
            IMG = cv2.imread(files[cnt])
            dst = cv2.resize(IMG, dsize=(0, 0), fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow('diagnosis')
            cv2.imshow('diagnosis', dst)
            cv2.moveWindow('diagnosis', 690, 120)
            siren_sound.play(0)
            label2.after(7000, label2.destroy)




        if cnt == 2:
            break

        cnt += 1

        time.sleep(2)



btn1 = Button(win, width=8, height=1, fg="white", text="진단 수행", command=diagnosis, bg="red")
btn1.place(x=200, y=690)

btn2 = Button(win, width=8, height=1, fg="black", text="과정 보기", font = "맑은고딕 12", command=show_label3, bg="white")
btn2.place(x=50, y=690)

btn3 = Button(win, width=9, height=1, fg="black", text="과정 숨기기", font = "맑은고딕 12", command=clear_photo_image, bg="white")
btn3.place(x=50, y=720)

orb.mainloop()
win.mainloop()
