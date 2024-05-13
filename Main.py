from tkinter import *
import os
#from PIL import Image, ImageTk
from tkinter import filedialog
from pymsgbox import *
from skimage.measure import compare_ssim
#import argparse
#import imutils

#from IPython.display import display
import math
import cv2
import numpy as np
from math import log10, sqrt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("original_image.png")
    compressed = cv2.imread("compressed_image.png", 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);

    return t;


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def endprogram():
    print("\nProgram terminated!")
    sys.exit()


def getImage():
    global df

    import_file_path = filedialog.askopenfilename()
    print(import_file_path)

    image = cv2.imread(import_file_path)
    print(image)
    filename = './image/test.jpg'
    cv2.imwrite(filename, image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)
    # import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im_invert.save('lena_invert.jpg', quality=95)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
    im_invert.save('horse_invert.png')
    image2 = cv2.imread('horse_invert.png')
    cv2.imshow("Invert", image2)

    """"-----------------------------------------------"""

    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image', img)
    cv2.imshow('Gray image', gray)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imshow("Nosie Removal", dst)
    src = cv2.imread(filename);

    I = src.astype('float64') / 255;

    dark = DarkChannel(I, 15);
    A = AtmLight(I, dark);
    te = TransmissionEstimate(I, A, 15);
    t = TransmissionRefine(src, te);
    J = Recover(I, t, A, 0.1);

    cv2.imshow("dark", dark);
    cv2.imshow("TransmissionEstimate", t);
    cv2.imshow('I', src);
    cv2.imshow('Enhance Image', J);
    cv2.imwrite("./image/J.png", J * 255);

    # Read images from file.
    original = cv2.imread("./image/test.jpg")
    compressed = cv2.imread("./image/J.png", 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")
    # imageA = cv2.imread(args["first"])
    # imageB = cv2.imread(args["second"])
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    cv2.waitKey();


def getvideo():
    remove_image_files('Data')
    remove_image_files('RData')
    global df

    import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    FrameCapture(import_file_path)


def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        try:
            cv2.imwrite("Data/%d.jpg" % count, image)



        except:
            print('Done')
       # cv2.imwrite(f'C:/Users/Fantasy-PC/PycharmProjects/NoiseRemovePy/Data/frame_{frameNr}.jpg', frame)

        count += 1
        cv2.waitKey();


import cv2
import os



folder_path = "Data/"


def noisere():
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)


            if image is not None:
                src = cv2.imread(img_path);
                I = src.astype('float64') / 255;
                dark = DarkChannel(I, 15);
                A = AtmLight(I, dark);
                te = TransmissionEstimate(I, A, 15);
                t = TransmissionRefine(src, te);
                J = Recover(I, t, A, 0.1);

                cv2.imwrite("./RData/"+filename, J * 255);


            else:
                print(f"Unable to read image: {filename}")

def extract_numbers(filename):
    return int(re.search(r'\d+', filename).group())


def images_to_video(folder_path, video_output_path, fps=25):
    # Get the list of image files in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No image files found in the folder.")
        return

    # Sort the image files by filename
    image_files.sort(key=extract_numbers)

    # Read the first image to get dimensions
    sample_image = cv2.imread(image_files[0])
    height, width, _ = sample_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Specify the video codec
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Write each image to the video file
    for image_file in image_files:
        print(image_file)
        img = cv2.imread(image_file)
        out.write(img)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved as {video_output_path}")

def display_video(video_path):
    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Display the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()




def Video():
    # Define paths
    folder_path = "RData"
    video_output_path = "output_video.mp4"

    # Convert images to video
    images_to_video(folder_path, video_output_path)

    # Display the video
    display_video(video_output_path)


def remove_image_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        # Check if it's a file and if it's an image file
        if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Remove the image file
                os.remove(file_path)
                print(f"Removed: {file}")
            except OSError as e:
                print(f"Error: {file} : {e.strerror}")


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.title("Defogging")

    Label(text="Defogging", bg="aqua", width="300", height="5", font=("Calibri", 16)).pack()

    Button(main_screen, text='''Upload Video''', font=('Verdana', 15), height="2", width="30", command=getvideo).pack()
    Label(text="").pack()
    Button(main_screen, text='''Noise Remove''', font=('Verdana', 15), height="2", width="30", command=noisere).pack()
    Label(text="").pack()

    Button(main_screen, text='''Video ''', font=('Verdana', 15), height="2", width="30", command=Video).pack()

    Label(text="").pack()

    Button(main_screen, text='''Upload Image''', font=('Verdana', 15), height="2", width="30", command=getImage).pack()

    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()
