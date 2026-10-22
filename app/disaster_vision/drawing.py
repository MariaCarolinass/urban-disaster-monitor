"""Draw matching RGB annotations for both model families."""

import cv2

from .config import CLASS_COLORS, FONT_COLORS


def _get_class_colors(class_name):
    name = str(class_name).strip().casefold()
    return CLASS_COLORS.get(name, (255, 255, 255)), FONT_COLORS.get(name, (0, 0, 0))


def _draw_bounding_boxes(image, annotations):
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    height, width = image.shape[:2]

    for box, class_name, confidence in annotations:
        x1, y1, x2, y2 = map(int, box)
        box_color, text_color = _get_class_colors(class_name)
        label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, min(x1, width - text_w - 2))
        text_y = y1 if y1 >= text_h + 3 else min(height - 1, y1 + text_h + 3)
        bg_tl = (text_x, text_y - text_h)
        bg_br = (text_x + text_w, text_y)

        cv2.rectangle(annotated_image, bg_tl, bg_br, box_color, -1)
        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y - 2),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 2)

    return annotated_image


def yolo_bounding_box(image, results):
    class_names = results[0].names
    annotations = (
        (box.xyxy[0], class_names[int(box.cls[0])], float(box.conf[0]))
        for box in results[0].boxes
    )
    return _draw_bounding_boxes(image, annotations)


def get_rfdetr_class_name(detections, index):
    data = getattr(detections, "data", None)
    if isinstance(data, dict):
        class_names = data.get("class_name")
        if class_names is not None:
            return str(class_names[index])

    class_ids = getattr(detections, "class_id", None)
    if class_ids is not None:
        return str(int(class_ids[index]))

    return "object"


def rfdetr_bounding_box(image, detections):
    boxes = getattr(detections, "xyxy", None)
    confidences = getattr(detections, "confidence", None)
    if boxes is None or len(boxes) == 0:
        return image.copy()
    annotations = (
        (box, get_rfdetr_class_name(detections, index),
         float(confidences[index]) if confidences is not None else None)
        for index, box in enumerate(boxes)
    )
    return _draw_bounding_boxes(image, annotations)
