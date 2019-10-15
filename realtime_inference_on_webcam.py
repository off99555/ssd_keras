from models.keras_ssd7 import build_model
from ssd_encoder_decoder.ssd_output_decoder import (
    decode_detections,
    decode_detections_fast,
)
import numpy as np
import cv2 as cv
from snoop import pp
from time import time

img_height = 144  # Height of the input images
img_width = 256  # Width of the input images
img_channels = 3  # Number of color channels of the input images
intensity_mean = (
    127.5
)  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = (
    127.5
)  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1  # Number of positive classes
scales = [
    0.08,
    0.16,
    0.32,
    0.64,
    0.96,
]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = (
    True
)  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = (
    None
)  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = (
    None
)  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = (
    False
)  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [
    1.0,
    1.0,
    1.0,
    1.0,
]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = (
    True
)  # Whether or not the model is supposed to use coordinates relative to the image size

model = build_model(
    image_size=(img_height, img_width, img_channels),
    n_classes=n_classes,
    mode="training",
    l2_regularization=0.0005,
    scales=scales,
    aspect_ratios_global=aspect_ratios,
    aspect_ratios_per_layer=None,
    two_boxes_for_ar1=two_boxes_for_ar1,
    steps=steps,
    offsets=offsets,
    clip_boxes=clip_boxes,
    variances=variances,
    normalize_coords=normalize_coords,
    subtract_mean=intensity_mean,
    divide_by_stddev=intensity_range,
)

model.load_weights("./ssd7_weights.h5", by_name=True)


def draw_inference(x):
    """Input and output will be an image in RGB format"""
    start = time()
    y_pred = model.predict(x[None])
    stop1 = time()
    print(y_pred.shape)

    y_pred_decoded = decode_detections(
        y_pred,
        confidence_thresh=0.3,
        iou_threshold=0.45,
        top_k=200,
        normalize_coords=normalize_coords,
        img_height=img_height,
        img_width=img_width,
    )
    stop2 = time()
    elapsed1 = (stop1 - start) * 1000
    elapsed2 = (stop2 - start) * 1000
    pp(elapsed1, elapsed2)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print("   class   conf xmin   ymin   xmax   ymax")
    print(y_pred_decoded[0])

    scale = 4
    # Draw the predicted boxes in blue
    for box in y_pred_decoded[0]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        color = (255, 0, 0)
        label = "{}: {:.0f} %".format("WMR", box[1] * 100)
        xmin = int(round(xmin) * scale)
        xmax = int(round(xmax) * scale)
        ymin = int(round(ymin) * scale)
        ymax = int(round(ymax) * scale)
        pp(xmin, ymin, xmax, ymax, label)
        x = cv.resize(x, (0, 0), None, fx=scale, fy=scale)
        x = cv.rectangle(x, (xmin, ymin), (xmax, ymax), color, 2)
        x = cv.putText(
            x, label, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
    return x


use_webcam = True
x = cv.imread(
    r"D:\off99555\Documents\ProgrammingProjects\DECA\MarkerBasedTracking\train_test_images\test\img00036.jpg",
    cv.IMREAD_GRAYSCALE,
)
x = cv.cvtColor(x, cv.COLOR_GRAY2RGB)
x = draw_inference(x)
x = cv.cvtColor(x, cv.COLOR_RGB2BGR)
cv.imshow("img", x)
cv.waitKey(0)
cv.destroyAllWindows()