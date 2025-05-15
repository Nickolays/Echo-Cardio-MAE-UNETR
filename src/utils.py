import numpy as np
from natsort import natsorted
import os
import cv2


def get_image_filepaths(main_path, format):
    
    # Сюда всё запишем
    filepaths = []
    # Итерируемся по всем пациентам
    for address, dirs, files in os.walk(main_path):
        # Все названия внутри каждого пациента
        for name in files:
            # Выбираем нужный формат
            if format in name:
                # Добавляем путь до изображения или маски
                if os.path.exists(os.path.join(address, name)):
                    filepaths.append(os.path.join(address, name))

    # assert len(files) == len(filepaths), f"There're another format of files, .txt, for instance. File is: {filepaths}"
    
    return natsorted(filepaths)

# Restore images to suitable images of opencv style
def ImgForPlot(img):
    # img = np.einsum('ijk->jki', img)
    # img = (127.5*(img+1)).astype(np.uint8)
    try:
        img = img.cpu().detach().numpy()
        return np.transpose(img, (1, 2, 0))
    except:
        print("Already numpy.array")
        return np.transpose(img, (1, 2, 0))

def get_mask(self, path):
    # Open .txt file
    try:
        txt = np.loadtxt(path)
    except:
        with open(path, 'r') as file:
            txt = file.readlines()
    cls_dict = self.prepare_txt(txt)

    masks = []
    for i in range(4):  # Because we have 4 classes
        mask = np.zeros((self.img_shape + (1, )))
        # Get coordinates for current class
        try:
            points = np.array([[[xi, yi]] for xi, yi in cls_dict[i+1]]).astype(np.int32)
            mask = cv2.fillPoly(mask, [points], color=[255, 255, 255])
            mask = mask / 255   # Scale
        except:
            mask = np.zeros((self.img_shape + (1, )))

        masks.append(mask.astype(np.float32))

    return np.concat(masks, axis=-1)

def prepare_txt(self, txt):
    """ Convert raw txt file to dict with {classes: np.array of points} """
    xy_dict = {}
    for string in txt:
        if isinstance(string, str):
            string = string.split(" ")
        cls = int(string[0])
        string = string[1:]  # Cut class
        # print(len(string))
        xy = []
        # Transform to (x, y) format
        for i in range(0, len(string), 2):
            x, y = string[i], string[i+1]
            if isinstance(x, str):
                x, y = float(x), float(y)
            if x < 1.0001:
                x, y = int(x * self.img_shape[0]), int(y * self.img_shape[1])
            xy.append((x, y))
        # Save
        xy_dict[cls] = np.array(xy)

    return xy_dict