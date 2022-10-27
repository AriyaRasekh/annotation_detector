from random import randrange

import cv2


class Bbox:
    all = []
    all_per_img = []
    #   l ----------------------
    #   |                      |
    #   |                      |
    #   |_____________________ r

    def __init__(self, l_x, l_y, r_x, r_y, TYPE):

        self.l_x, self.l_y, self.r_x, self.r_y = l_x, l_y, r_x, r_y
        self.TYPE = TYPE  # 1: text
                          # 2: drawing
        self.BOX_COLOR = (255, 0, 0) if self.TYPE == 1 else (0, 255, 0)
        Bbox.all.append(self)
        Bbox.all_per_img.append(self)

    @staticmethod
    def do_overlap(l, r):
        """
        :param l: x and y coordinates of l as (x, y)
        :param r: x and y coordinates of r as (x, y)
        :return: true if there is an overlap, false otherwise
        """

        if len(Bbox.all_per_img) == 0:
            return False

        for box in Bbox.all_per_img:

            # If one rectangle is on left side of other
            if l[0] > box.r_x or box.l_x > r[0]:
                continue
            # If one rectangle is above other
            elif l[1] > box.r_y or box.l_y > r[1]:
                continue

            else:
                return True

        return False

    @staticmethod
    def random_bbox_location(image, text_box):
        """return bbox coordinate which text box fits in the image"""
        text_x = randrange(image.shape[1] - text_box.shape[1])
        text_y = randrange(image.shape[0] - text_box.shape[0])
        l = (text_x, text_y)
        r = (text_x + text_box.shape[1], text_y + text_box.shape[0])
        return l, r

    @staticmethod
    def draw_bboxes(image):
        for box in Bbox.all_per_img:
            BOX_COLOR = box.BOX_COLOR
            cv2.rectangle(image, (box.l_x, box.l_y), (box.r_x, box.r_y), BOX_COLOR, 1)

    @staticmethod
    def draw_last_bbox(image):
        BOX_COLOR = Bbox.all[-1].BOX_COLOR
        cv2.rectangle(image, (Bbox.all[-1].l_x, Bbox.all[-1].l_y), (Bbox.all[-1].r_x, Bbox.all[-1].r_y), BOX_COLOR, 1)

    @staticmethod
    def get_annotation_info_yoloV5(image_width, image_height):
        annotation_image_data = ""
        for bounding_box in Bbox.all_per_img:
            box_width = bounding_box.r_x - bounding_box.l_x
            box_height = bounding_box.r_y - bounding_box.l_y

            x_center = bounding_box.l_x + (box_width/2)
            y_center = bounding_box.l_y + (box_height/2)

            annotation_image_data += f"{bounding_box.TYPE} {x_center/image_width} {y_center/image_height} {box_width/image_width} {box_height/image_height}\n"

        return annotation_image_data



    def __repr__(self):
        print(f"self.l_x: {self.l_x}, self.l_y: {self.l_y}, self.r_x: {self.r_x}, self.r_y: {self.r_y}")
