from typing import List, Union, Tuple

import numpy as np

#from skimage.transform import resize as imresize


def normalise_bbs_to_top_left_right_bottom(bounding_boxes: np.ndarray,
                                           image_shapes: List[Union[List[int], Tuple[int]]] = None,
                                           bb_uses_wh: bool = False,
                                           bb_use_xy_as_center: bool = False,
                                           convert_to_relative: bool = False):
    """
    Normalise the bbs to be [x_left, y_top, x_right, y_bottom]. Notice that a batch of
    bounding boxes can be passed, the first dimension is considered to be the batch size. So, the images_shapes
    is a list with the dimensions of each individual size.
    :param bounding_boxes: This is an array where the first dimension is the batch size (different images) and the last
                            one is the components of the bounding box so 4 values. The intermidiate dimensions are not
                            relevant.
    :param bb_uses_wh: Whether the last two elements of the bb are the width and height or the bottom right point
    :param bb_use_xy_as_center: Whether the first two elements of the bb are the center location or the
                                top left point
    :param convert_to_relative: Whether the bb output should be relative values with respect to image size
                                (values from 0 to 1) or the real values (values from 0 to image size)
    :param image_shapes: As many as the batch size. This is a list with as many as batch size and each containing a
                        tuple with 2 or 3 values like the ones returned by image.shape, being image an np array.

    :return: None
    """
    bounding_boxes_output = bounding_boxes.copy()
    if bb_uses_wh and bb_use_xy_as_center:
        bounding_boxes_output = np.concatenate([bounding_boxes[..., :2] - bounding_boxes[..., 2:4] * 0.5,
                                   bounding_boxes[..., :2] + bounding_boxes[..., 2:4] * 0.5], axis=-1)
    elif bb_uses_wh:
        bounding_boxes_output[..., 2:4] = bounding_boxes[..., :2] + bounding_boxes[..., 2:4]
    elif bb_use_xy_as_center:
        bounding_boxes_output[..., :2] = 2 * bounding_boxes[..., :2] - bounding_boxes[..., 2:4]

    if convert_to_relative:
        bounding_boxes_output = convert_absolute_to_relative_size(bounding_boxes_output, image_shapes)

    return bounding_boxes_output


def convert_absolute_to_relative_size(bounding_boxes: np.ndarray, image_shapes: List[Union[List[int], Tuple[int]]]):
    """
    Convert absolute bounding box positions to relative ones by dividing by the image shape. Notice that a batch of
    bounding boxes can be passed, the first dimension is considered to be the batch size. So, the images_shapes
    is a list with the dimensions of each individual size.
    :param bounding_boxes: This is an array where the first dimension is the batch size (different images) and the last
                            one is the components of the bounding box so 4 values. The intermediate dimensions are not
                            relevant.
    :param image_shapes: As many as the batch size
    :return:
    """
    bounding_boxes_output = bounding_boxes.copy()
    if np.max(bounding_boxes) > 1:
        bounding_boxes_output = bounding_boxes_output.astype(np.float32)
        if isinstance(image_shapes, (tuple, list)):
            if isinstance(image_shapes[0], (int)):
                image_shapes = [image_shapes]
        else:
            raise TypeError('image_shapes must be a list or tuple')

        if len(image_shapes) != bounding_boxes.shape[0]:
            ValueError('The number of different images in the first dimension of bounding boxes must be the same',
                       ' as the number of image shapes')
        for i, im_shape in enumerate(image_shapes):
            h, w = im_shape[:2]
            size = np.array([w, h, w, h]).astype(np.float32).reshape(1, -1)
            bounding_boxes_output[i, ...] = bounding_boxes[i, ...] / size

    return bounding_boxes_output


def remove_ambiguous_classes(*,
                             image: np.ndarray,
                             ambiguous_classes: Union[str, int, List[Union[str, int]]],
                             bbs: Union[np.ndarray, List[List[Union[float, int]]]] = None,
                             bbs_classes: Union[np.ndarray, List[Union[str, int]]] = None,
                             mask: np.ndarray = None,
                             intensity_range=(0, 256),
                             semantic_channel=1,
                             xywh=False,
                             return_mask=False):

    """
    Change the content of bounding boxes with labels in ambiguous_classes as noise. If there is an overlapping with
    bounding boxes that are not in the ambiguous_classes then the overlapping part is not changed.
    :param image: The image
    :param bbs: A set of bounding boxes for the image, they can be relative to the size of the image or the real values
                the bounding boxes are [left, top, right, bottom]
    :param bbs_classes: A list with the label of each bounding box. It can be integers or string as long as
                        ambiguous_classes are of the same type
    :param mask: Masks with the semantic labels, instance labels and all the information. The channel with the
                    semantic labels can be passed as semantic_channel
    :param semantic_channel: The channel in mask containing the labels that needs to be checked.
    :param ambiguous_classes: A list with the labels that are going to be removed. It can be integers or string
                                as long as bbs_classes are of the same type
    :param intensity_range: The range of the intensities for the random allocation, by default to 256
    :param xywh: When True the boxes are assumed to be center, width, height instead of xleft, ytop, xright, ybottom
    :param return_mask: Return the mask to remove ambiguous classes
    :return: The corrected image and the bbs and bbs_classes without the ambiguous_classes
    """

    if isinstance(ambiguous_classes, (np.ndarray, list, tuple)):
        ambiguous_classes = np.array(ambiguous_classes)
    else:
        raise TypeError('ambiguous_classes are of an unknown classes')

    n_channels = 1
    if len(image.shape) > 2 and image.shape[2] > 1:
        n_channels = image.shape[2]

    output_image = image.copy()
    boxes_not_amb = []
    outputs = []
    if bbs is not None:
        if bbs_classes is None:
            raise TypeError('bbs_class cannot be None if bbs is not None')

        if isinstance(bbs_classes, (str, int)):
            bbs_classes = [bbs_classes]

        if isinstance(bbs_classes, (np.ndarray, list, tuple)):
            bbs_classes = np.array(bbs_classes)
        else:
            raise TypeError('ambiguous_classes are of an unknown classes')

        if isinstance(bbs, (np.ndarray, list)):
            bbs = np.array(bbs)
        else:
            raise TypeError('ambiguous_classes are of an unknown classes')

        if len(bbs) != len(bbs_classes):
            raise ValueError('The number of elements in bbs and in bbs_classes must be the same')

        mask_ambiguous = np.isin(bbs_classes, ambiguous_classes)
        bbs_classes_no_ambiguous = bbs_classes[~mask_ambiguous]
        bbs_no_ambiguous_out = bbs[~mask_ambiguous]
        if np.all(bbs <= 1.0):
            h, w = image.shape[:2]
            bbs = bbs * np.array([w, h, w, h]).reshape([1, -1])

        bbs = np.floor(bbs).astype(int)
        bbs_no_ambiguous = bbs[~mask_ambiguous]
        bbs_ambiguous = bbs[mask_ambiguous]

        if xywh:
            boxes_not_amb = normalise_bbs_to_top_left_right_bottom(bbs_no_ambiguous, bb_use_xy_as_center=True, bb_uses_wh=True)
            boxes_not_amb[boxes_not_amb < 0] = 0
            boxes_not_amb = np.floor(boxes_not_amb).astype(int)
            boxes_amb = normalise_bbs_to_top_left_right_bottom(bbs_ambiguous, bb_use_xy_as_center=True, bb_uses_wh=True)
            boxes_amb[boxes_amb < 0] = 0
            boxes_amb = np.floor(boxes_amb).astype(int)
        else:
            boxes_not_amb = bbs_no_ambiguous
            boxes_amb = bbs_ambiguous

        for bb in boxes_amb:
            bb = np.maximum(0, bb)
            h, w = output_image.shape[:2]
            bb = np.minimum(np.array([w - 1, h - 1, w - 1, h - 1]), bb)
            output_image[bb[1]:bb[3], bb[0]:bb[2], ...] = np.random.randint(intensity_range[0], intensity_range[1],
                                                                            [bb[3]-bb[1], bb[2]-bb[0], n_channels])

        outputs.extend([bbs_no_ambiguous_out, bbs_classes_no_ambiguous])

    if mask is not None:
        mask_output = np.zeros_like(mask)
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            semantic_mask = mask
        else:
            semantic_mask = mask[..., semantic_channel]
        values = np.unique(semantic_mask)
        mask_not_ambiguous = values[~np.isin(values, ambiguous_classes)]
        mask_ambiguous = values[np.isin(values, ambiguous_classes)]
        random_image = np.random.randint(intensity_range[0], intensity_range[1], output_image.shape)
        for val in mask_ambiguous:
            output_image[semantic_mask == val, ...] = random_image[semantic_mask == val, ...]

        for val in mask_not_ambiguous:
            output_image[semantic_mask == val, ...] = image[semantic_mask == val, ...]
            mask_output[semantic_mask == val, ...] = mask[semantic_mask == val, ...]

        outputs.append(mask_output)

    for bb in boxes_not_amb:
        output_image[bb[1]:bb[3], bb[0]:bb[2], ...] = image[bb[1]:bb[3], bb[0]:bb[2], ...]

    outputs.insert(0, output_image)

    if return_mask:
        outputs.append(~mask_ambiguous)

    return tuple(outputs)
