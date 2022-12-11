DATA_SET_NAME = 'Chest_xRay_1'
VALIDATION_SIZE = 0.2  # validation data-size train and validation split, 0.2 means 20% validation, 80% training
NUMBER_OF_SYNTHETIC_IMAGES = 800  # total number of synthetic images to be created
WHITENESS_THRESHOLD = 205  # any pixel value > WHITENESS_THRESHOLD wil be considered as background for handwritten words
NUM_PRINTED_TEXT = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5]  # possible number of typed word on each image
text_background_probability = 0.1  # between 0 and 1, the probability for a word to have text background
NUM_HANDWRITTEN_WORDS = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5]  # possible number of hand written words on each image
AVAILABLE_FONTS = [0, 1, 2, 3]  # available fonts in Open-CV
                                # 0: FONT_HERSHEY_SIMPLEX
                                # 1: FONT_HERSHEY_PLAIN
                                # 2: FONT_HERSHEY_DUPLEX
                                # 3: FONT_HERSHEY_COMPLEX
                                # 4: FONT_HERSHEY_TRIPLEX
                                # 5: FONT_HERSHEY_COMPLEX_SMALL
                                # 6: FONT_HERSHEY_SCRIPT_SIMPLEX
                                # 7: FONT_HERSHEY_SCRIPT_COMPLEX
AVAILABLE_COLORS = [(255, 255, 255),    # possible word colors in RGB format
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255)]
AVAILABLE_THICKNESS = [1, 2]
AVAILABLE_LINE_TYPES = [2]
AVAILABLE_SIZES = [75, 100, 100, 100, 150, 200, 300]
ANGELS = [0, 0, 0, 45, 90, 180, 270, 315]
