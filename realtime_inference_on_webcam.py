from models.keras_ssd7 import build_model
from ssd_encoder_decoder.ssd_output_decoder import (
    decode_detections,
    decode_detections_fast,
)
import numpy as np
import cv2 as cv
from snoop import pp

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

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

x = cv.imread(
    r"D:\off99555\Documents\ProgrammingProjects\DECA\MarkerBasedTracking\train_test_images\test\img00036.jpg"
)
x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
y_pred = model.predict(x[None])
print(y_pred.shape)
# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

y_pred_decoded = decode_detections(
    y_pred,
    confidence_thresh=0.3,
    iou_threshold=0.45,
    top_k=200,
    normalize_coords=normalize_coords,
    img_height=img_height,
    img_width=img_width,
)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print("   class   conf xmin   ymin   xmax   ymax")
print(y_pred_decoded[0])

# Draw the predicted boxes in blue
for box in y_pred_decoded[0]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = (1, 0, 0)
    label = "{}: {:.2f}".format("WMR", box[1])
    xmin = int(round(xmin))
    xmax = int(round(xmax))
    ymin = int(round(ymin))
    ymax = int(round(ymax))
    pp(xmin, ymin, xmax, ymax, label)
    x = cv.rectangle(x, (xmin, ymin), (xmax, ymax), color)
    x = cv.cvtColor(x, cv.COLOR_RGB2BGR)
    cv.imshow("img", x)
    cv.waitKey(0)
    # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
cv.destroyAllWindows()
