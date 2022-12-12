import copy
import os
import numpy as np
import cv2
import random
import requests
import pickle
import splitfolders
import config
import data_genetor_config
from bbox import Bbox

DEBUG = False
SHOW_SAMPLE_OUT_PUT_MAX = 0


class MedIMG:
    WHITENESS_THRESHOLD = data_genetor_config.WHITENESS_THRESHOLD  # any pixel value > WHITENESS_THRESHOLD wil be considered as background for handwritten words
    WORD_DATA_BASE = config.WORD_DATA_BASE
    HANDWRITTEN_WORDS_PATH = config.HANDWRITTEN_WORDS_PATH
    HANDWRITTEN_WORDS_IDS_PATH = config.HANDWRITTEN_WORDS_IDS_PATH
    NUM_PRINTED_TEXT = data_genetor_config.NUM_PRINTED_TEXT
    NUM_HANDWRITTEN_WORDS = data_genetor_config.NUM_HANDWRITTEN_WORDS

    def __init__(self, PATH):
        self.OVERLAP = False

        response = requests.get(MedIMG.WORD_DATA_BASE)
        self.WORDS = response.content.splitlines()

        with open(MedIMG.HANDWRITTEN_WORDS_IDS_PATH, 'rb') as f:
            self.HANDWRITTEN_WORDS_IDS = pickle.load(f)

        for counter, i in enumerate(self.WORDS):
            self.WORDS[counter] = i.decode("utf-8")

        self.image = self.load_from_path(PATH)
        num_of_texts = MedIMG.NUM_PRINTED_TEXT[random.randrange(len(MedIMG.NUM_PRINTED_TEXT))]
        num_of_words = MedIMG.NUM_HANDWRITTEN_WORDS[random.randrange(len(MedIMG.NUM_HANDWRITTEN_WORDS))]

        for i in range(num_of_texts):
            self.generate_text()

        for i in range(num_of_words):
            random_handwritten_word = self.HANDWRITTEN_WORDS_IDS[random.randrange(len(self.HANDWRITTEN_WORDS_IDS))]
            self.generate_handwritten_text(MedIMG.HANDWRITTEN_WORDS_PATH + random_handwritten_word)

    @staticmethod
    def draw_text(img, text,
                  font,
                  pos,
                  font_scale,
                  font_thickness,
                  text_color,
                  text_color_bg,
                  line_type
                  ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y - text_h), text_color_bg, -1)
        cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, line_type)

    def generate_text(self):
        out_raw = np.zeros((700, 700, 3), np.uint8)
        x_initial, y_initial = 200, 300
        word_info = self.generate_random_word_info()
        FONTSCALE = 1
        text_width, text_height = cv2.getTextSize(word_info["WORD"], word_info["FONT"], FONTSCALE, word_info["LINE_TYPE"])[0]
        if x_initial + text_width > out_raw.shape[1] or y_initial + text_height > out_raw.shape[0]:
            print("returning None 67")
            return None
        x_c, y_c = int(x_initial + text_width / 2), int(y_initial - text_height / 2)
        fill_background = True if random.random() < data_genetor_config.text_background_probability else False
        # fill_background = Falbse
        newX, newY, newX2, newY2, M = MedIMG.get_rotated_points(x_c, y_c, word_info["ANGLE"], text_height, text_width)

        if fill_background:
            MedIMG.draw_text(out_raw,
                             word_info["WORD"],
                             font=word_info["FONT"],
                             font_scale=FONTSCALE,
                             font_thickness=word_info["THICKNESS"],
                             pos=(x_initial, y_initial),
                             text_color=word_info["COLOR"],
                             text_color_bg=(123, 123, 123),
                             line_type=word_info["LINE_TYPE"])


        else:
            cv2.putText(out_raw,
                        word_info["WORD"],
                        (x_initial, y_initial),
                        word_info["FONT"],
                        FONTSCALE,
                        word_info["COLOR"],
                        word_info["THICKNESS"],
                        word_info["LINE_TYPE"])

        out_raw = cv2.warpAffine(out_raw, M, (out_raw.shape[1], out_raw.shape[0]))
        text_box = out_raw[newY:newY2, newX:newX2, :]

        # bbox location which text is watermarked on image
        l, r = Bbox.random_bbox_location(self.image, text_box)

        if not self.OVERLAP:
            MAX_ALLOWED_TRIES = 100
            tries_counter = 0
            while Bbox.do_overlap(l, r):
                if tries_counter > MAX_ALLOWED_TRIES:
                    print("returning None 105")
                    return None
                # get another random location
                l, r = Bbox.random_bbox_location(self.image, text_box)
                tries_counter += 1

        self.watermark_word(text_box, l[0], l[1], fill_background=fill_background)

        Bbox(l[0], l[1], r[0], r[1], TYPE=0)  # adding bbox

    def watermark_word(self, word, x, y, fill_background=False):
        if fill_background:
            for i, row in enumerate(word):
                for j, pixel in enumerate(row):
                    if pixel[0] == 123 and pixel[1] == 123 and pixel[2] == 123:
                        self.image[y + i, x + j] = [0, 0, 0]

        else:
            for i, row in enumerate(word):
                for j, pixel in enumerate(row):
                    if np.amax(pixel) > 0:
                        self.image[y + i, x + j] = [pixel[2], pixel[1], pixel[0]]

    def watermark_handwritten_word(self, word, x, y, color):
        for i, row in enumerate(word):
            for j, pixel in enumerate(row):
                if np.amax(pixel) < MedIMG.WHITENESS_THRESHOLD:
                    self.image[y + i, x + j] = [color[0], color[1], color[2]]

    def generate_random_word_info(self, handwritten=False):

        AVAILABLE_FONTS = data_genetor_config.AVAILABLE_FONTS
        AVAILABLE_COLORS = data_genetor_config.AVAILABLE_COLORS
        AVAILABLE_THICKNESS = data_genetor_config.AVAILABLE_THICKNESS
        AVAILABLE_LINE_TYPES = data_genetor_config.AVAILABLE_LINE_TYPES
        AVAILABLE_SIZES = data_genetor_config.AVAILABLE_SIZES
        ANGELS = data_genetor_config.ANGELS

        ANGLE = ANGELS[random.randrange(len(ANGELS))]
        FONT = AVAILABLE_FONTS[random.randrange(len(AVAILABLE_FONTS))]
        COLOR = AVAILABLE_COLORS[random.randrange(len(AVAILABLE_COLORS))]

        if handwritten:
            # ONLY hand-written word specifics
            RESIZE = AVAILABLE_SIZES[random.randrange(len(AVAILABLE_SIZES))]
            COLOR = (random.randrange(255), random.randrange(255), random.randrange(255))
            word_info = {
                "ANGLE": ANGLE,
                "COLOR": COLOR,
                "Resize_value": RESIZE  # re-sizes the original picture by Resize_value%
            }

        else:
            # ONLY printed word specifics
            THICKNESS = AVAILABLE_THICKNESS[random.randrange(len(AVAILABLE_THICKNESS))]
            LINE_TYPE = AVAILABLE_LINE_TYPES[random.randrange(len(AVAILABLE_LINE_TYPES))]
            WORD = self.WORDS[random.randrange(len(self.WORDS))]
            if DEBUG: print(f"picked word is {WORD}")

            word_info = {
                "ANGLE": ANGLE,
                "FONT": FONT,
                "COLOR": COLOR,
                "THICKNESS": THICKNESS,
                "LINE_TYPE": LINE_TYPE,
                "WORD": WORD
            }

        return word_info

    def generate_handwritten_text(self, data_path):

        out_raw = np.zeros((700, 700, 3), np.uint8)
        out_raw.fill(255)  # creating a white raw picture
        word_img = self.load_from_path(data_path)
        x_initial, y_initial = 100, 200

        word_info = self.generate_random_word_info(handwritten=True)

        scale_width = int(word_img.shape[1] * word_info["Resize_value"] / 100)
        scale_height = int(word_img.shape[0] * word_info["Resize_value"] / 100)
        dim = (scale_width, scale_height)
        word_img = cv2.resize(word_img, dim, interpolation=cv2.INTER_AREA)

        text_height, text_width = word_img.shape[0], word_img.shape[1]
        x_c, y_c = int(x_initial + text_width / 2), int(y_initial + text_height / 2)
        if DEBUG:    print(f"data path: {data_path} | images shape: {word_img.shape} | {out_raw.shape}")
        if x_initial + text_width > out_raw.shape[1] or y_initial + text_height > out_raw.shape[0]:
            print("returning None 207")
            return None
        out_raw[y_initial:y_initial + text_height, x_initial:x_initial + text_width] = word_img

        newX, newY, newX2, newY2, M = MedIMG.get_rotated_points(x_c, y_c, word_info["ANGLE"], text_height, text_width)

        out_raw = cv2.warpAffine(out_raw, M, (out_raw.shape[1], out_raw.shape[0]))

        text_box = out_raw[newY:newY2, newX:newX2, :]

        # bbox location which text is watermarked on image
        l, r = Bbox.random_bbox_location(self.image, text_box)

        if not self.OVERLAP:
            MAX_ALLOWED_TRIES = 100
            tries_counter = 0
            while Bbox.do_overlap(l, r):
                if tries_counter > MAX_ALLOWED_TRIES:
                    print("returning None 224")
                    return None
                # get another random location
                l, r = Bbox.random_bbox_location(self.image, text_box)
                tries_counter += 1

        self.watermark_handwritten_word(text_box, l[0], l[1], word_info["COLOR"])

        Bbox(l[0], l[1], r[0], r[1], TYPE=0)  # adding bbox         TYPE is the object label

    @staticmethod
    def load_from_path(PATH):
        image = cv2.imread(f"{PATH}", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError

        return image

    @staticmethod
    def get_rotated_points(x_c, y_c, angle, text_height, text_width):
        M = cv2.getRotationMatrix2D((x_c, y_c), angle, 1)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        newW = int((text_height * sin) + (text_width * cos))
        newH = int((text_height * cos) + (text_width * sin))
        newX = x_c - int(newW / 2)
        newY = y_c - int(newH / 2)
        newX2 = x_c + int(newW / 2)
        newY2 = y_c + int(newH / 2)

        return newX, newY, newX2, newY2, M

    def save_img(self, full_path):
        writeStatus = cv2.imwrite(full_path, self.image)
        return writeStatus


class DataGenerator:

    def __init__(self, row_img):

        self.no_bbox_img = copy.deepcopy(row_img)
        self.bbox_img = copy.deepcopy(row_img)
        Bbox.draw_bboxes(self.bbox_img)
        self.drawing = False

        self.line_color = None
        self.line_thickness = None
        self.last_x = None
        self.last_y = None
        self.make_random_line()

        # to store bbox
        self.l_x = None
        self.l_y = None
        self.r_x = None
        self.r_y = None

        # to store shape's initial x and y
        self.shape_initial_x = None
        self.shape_initial_y = None

    def draw_line(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drawing = True
            cv2.line(self.no_bbox_img, (x, y), (x, y), self.line_color, thickness=self.line_thickness)
            cv2.line(self.bbox_img, (x, y), (x, y), self.line_color, thickness=self.line_thickness)
            self.last_x, self.last_y = x, y

            # only initial when drawing the first point of the shape
            if self.l_x is None:
                self.l_x = x
                self.l_y = y
                self.r_x = x
                self.r_y = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.no_bbox_img, (self.last_x, self.last_y), (x, y), self.line_color,
                         thickness=self.line_thickness)
                cv2.line(self.bbox_img, (self.last_x, self.last_y), (x, y), self.line_color,
                         thickness=self.line_thickness)
                self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.no_bbox_img, (self.last_x, self.last_y), (x, y), self.line_color,
                     thickness=self.line_thickness)
            cv2.line(self.bbox_img, (self.last_x, self.last_y), (x, y), self.line_color, thickness=self.line_thickness)

        if self.drawing:
            if x < self.l_x:
                self.l_x = x

            if x > self.r_x:
                self.r_x = x

            if y < self.l_y:
                self.l_y = y

            if y > self.r_y:
                self.r_y = y

    def save_img(self):
        cv2.imwrite("savedImage.png", self.no_bbox_img)

    def make_random_line(self):
        LINE_COLORS = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0)]
        THICKNESS_MAX = 10
        self.line_color = LINE_COLORS[random.randrange(len(LINE_COLORS))]
        self.line_thickness = random.randint(1, THICKNESS_MAX)

    def add_to_bbox(self):

        if self.l_x is None or self.l_y is None or self.r_x is None or self.r_y is None:
            print("ERROR: could no detect any drawing...")

        else:
            bbox_dimension = (self.l_x - (self.line_thickness // 2 + 1),
                              self.l_y - (self.line_thickness // 2 + 1),
                              self.r_x + (self.line_thickness // 2 + 1),
                              self.r_y + (self.line_thickness // 2 + 1))

            Bbox(bbox_dimension[0], bbox_dimension[1], bbox_dimension[2], bbox_dimension[3], TYPE=2)
            Bbox.draw_last_bbox(self.bbox_img)
            self.l_x = None
            self.l_y = None
            self.r_x = None
            self.r_y = None


def plot_bounding_box(image_file_path, annotation_file_path):
    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    with open(annotation_file_path, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]
        # print(annotation_list)

    w, h = image.shape[1], image.shape[0]
    annotations = np.array(annotation_list)
    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

    cv2.imshow('image window', image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    SHOW_BBOX = True
    NUMBER_OF_SYNTHETIC_IMAGES = data_genetor_config.NUMBER_OF_SYNTHETIC_IMAGES
    X_RAY_SCAN_IDS_PATH = config.X_RAY_SCAN_IDS_PATH
    OUTPUT_PATH = config.DATA_OUTPUT
    DATASET_NAME = data_genetor_config.DATA_SET_NAME
    IMG_OUTPUT = f"{OUTPUT_PATH}{DATASET_NAME}\\images\\"
    LABELS_OUTPUT = f"{OUTPUT_PATH}{DATASET_NAME}\\labels\\"

    image_ids = []
    annotation_data = []

    # creating dataset.yaml file containing dataset meta-data
    dataset_meta_data = f"path: {config.DATA_OUTPUT}{DATASET_NAME}  # dataset root dir\n" \
                        "train: images\\train\\class1  # train images (relative to 'path')\n" \
                        "val: images\\val\\class1  # val images (relative to 'path')\n" \
                        "test: \n" \
                        "names:\n" \
                        "  0: annotation\n"
    with open(f"{config.DATASET_METADATA_PATH + DATASET_NAME}.yaml", "w") as text_file:
        text_file.write(dataset_meta_data)

    if not os.path.exists(f"{OUTPUT_PATH}{DATASET_NAME}"):
        os.makedirs(f"{OUTPUT_PATH}{DATASET_NAME}")
    if not os.path.exists(IMG_OUTPUT):
        os.makedirs(IMG_OUTPUT)
    if not os.path.exists(f"{IMG_OUTPUT}class1"):
        os.makedirs(f"{IMG_OUTPUT}class1")
    if not os.path.exists(f"{LABELS_OUTPUT}class1"):
        os.makedirs(f"{LABELS_OUTPUT}class1")
    if not os.path.exists(IMG_OUTPUT):
        os.makedirs(IMG_OUTPUT)
    if not os.path.exists(LABELS_OUTPUT):
        os.makedirs(LABELS_OUTPUT)

    with open(X_RAY_SCAN_IDS_PATH, 'rb') as f:
        x_ray_ids = pickle.load(f)

    training_pic_id = 0
    verification_pic_id = 0
    training_testing_ids = []
    verification_ids = []

    pic_id = 0
    sample_out_put_counter = 0
    generated_image_counter = 0
    while generated_image_counter < NUMBER_OF_SYNTHETIC_IMAGES:

        for id in x_ray_ids:
            if generated_image_counter >= NUMBER_OF_SYNTHETIC_IMAGES:
                break

            image_name = f"{str(pic_id).zfill(5)}.jpg"
            med_scan = MedIMG(f"{config.X_RAY_SCAN_PATH}{id}")
            writeStatus = med_scan.save_img(f"{IMG_OUTPUT}class1\\test_{image_name}")

            if writeStatus:

                annotation_image_data = Bbox.get_annotation_info_yoloV5(med_scan.image.shape[1], med_scan.image.shape[0])
                with open(f"{LABELS_OUTPUT}class1\\test_{str(pic_id).zfill(5)}.txt", "w") as text_file:
                    text_file.write(annotation_image_data)

                if sample_out_put_counter < SHOW_SAMPLE_OUT_PUT_MAX:
                    plot_bounding_box(f"{IMG_OUTPUT}{image_name}", f"{LABELS_OUTPUT}{str(pic_id).zfill(5)}.txt")
                    sample_out_put_counter += 1

                image_ids.append(image_name)
                pic_id += 1
                print(f"{image_name} successfully created.")
                generated_image_counter += 1


            else:
                print(f"ERROR: can't write image {image_name} ")

            Bbox.all_per_img = []

    # partition data-set, lucking the seed ensure same text files are copied in the corresponding directories
    splitfolders.ratio(IMG_OUTPUT, output=IMG_OUTPUT, ratio=(1-data_genetor_config.VALIDATION_SIZE, data_genetor_config.VALIDATION_SIZE), move=True)
    splitfolders.ratio(LABELS_OUTPUT, output=LABELS_OUTPUT, ratio=(1-data_genetor_config.VALIDATION_SIZE, data_genetor_config.VALIDATION_SIZE), move=True)
    os.rmdir(f"{IMG_OUTPUT}class1")
    os.rmdir(f"{LABELS_OUTPUT}class1")
