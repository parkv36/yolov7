from bisect import bisect_left
from functools import partial
import random
import secrets
from typing import List

import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader


def get_indices(indices=None, labels_index=None, index=None, num_indices=1,
                probabilities=None, non_weighted=True, map=None):
    """

    """
    if non_weighted:
        if num_indices > 1:
            return [indices[index]] + random.choices(indices, k=num_indices - 1)
        return [indices[index]]
    else:
        indices = []
        if not probabilities:
            probabilities = {key: float(i+1)/len(labels_index) for i, key in enumerate(labels_index.keys())}
        values = np.array(list(probabilities.values()))
        keys = list(probabilities.keys())
        if map:
            keys = [map[key] for key in keys]
        pos = np.searchsorted(values/values[-1], np.random.rand(num_indices))
        for pos_i in pos:
            all_files = labels_index[keys[pos_i]]
            indices.append(all_files[np.random.randint(0, len(all_files), 1)[0]])
        return indices

class DataGenerator(Dataset):
    BASE = 'pytorch'
    DEFAULT_LOADER_CONFIGURATION = {}

    def __init__(self, list_data: List[tuple],
                 batch_size=32, dim=(32, 32), shuffle=True, num_iter=1000, shuffle_folders=True, create_data=False,
                 class_prob: list = None, fix_iterations=False, num_glimpses=0, glimpse_size=(32, 32),
                 preprocessing_function=None, group_images=1, group_probability=0, self_supervision=False,
                 increase_y_size_for_loss=True, get_data_path=None, output_as_numpy=True, name='generator',
                 multilabel=None, annotations_are_masks=False, excluded_labels=None, default_label='negative'):


        """
        Create a generator for a given deep learning library.
        :param list_data:  A list of lists or tuples where the first value is the label or mask address if segmentation
                            or caption if captioning and the rest of the values are the image and whatever is required. For instance,
                            we could have a list with (label, image_filename, bounding_box)
        :param batch_size: int with batch size
        :param dim: A tuple or list with the size of the image (no channel). This is only useful in the case of using
                    glimpses. Otherwise, the size of the image i not relevant since all the operations are performed
                    in the preprocessing function.
        :param shuffle: whether to shuffle the data or not
        :param num_iter: The number of iterations in one epoch, this is used when either when shuffle_folder
                                or fix_iterations are set to True. Otherwise, it will use all the available data.
        :param shuffle_folders (boolean): It will select randomly a label and then randomly a data with that label,
                                this is useful to balance out data, since this does not allow epochs, num_iter
                                is used. Notice that this should not be used in the case of segmentation.
        :param class_prob: (list floats) The idea is to give to each folder a probability when they should not be
                            equally likely. The sum of all the values should be 1.
        :param fix_iterations:  When True the number of iterations is set to num_iter. This is not used when
                                shuffle_folders = True, since it cannot use all the data, so it shuffle_foler=True
                                implies fix_iterations = True regardless of the user input.
        :param num_glimpses:  This is the number of subimages to split an image. The idea is to break the image into
                            parts that are going to be used in the batch size to be able to pass through CNNs and the
                            idea is to avoid resizing the image
        :param glimpse_size: The size of each subimage, it must contain row size and column size and cannot
                                    be larger than the size of the image, it should around 1/4 or so in area.
        :param preprocessing_function: A function to perform on the image
        :param group_images: The number of images to group before sending them to the preprocessing function
        :param group_probability: The probability of grouping the images.
        :param self_supervision (boolean): When True, the generator assumes the pre-processing function will do some self-supervision
                                and return a label
        :param increase_y_size_for_loss: Most of the times the losses require the labels to be at least batch size
                                        x num_classes even when the number of classes is 1, we need a matrix
                                        batch_size x 1 instead of a vector. This is activated by default, if it
                                        is not necessary the flag can be turned off.
        :param get_data_path: This generator works with a list of data where the first value is the label and the
                            rest are all the parameters needed to create the input data and output to train a model
                            This function will extract the filename or will create a unique identifier for each
                            sample using the values from the input list. The identifier should be something that
                            it can allow to retrive the data. This is used in order to store the latest sampels that
                            are used, which can be retreive as last_image_paths. By default, lambda x: x[0] is used
        :param annotations_are_masks: When True it is assumed that y and y_sup are masks. This is only used when
                                        the base is pytorch in order to transpose the channels and H.
        :param output_as_numpy: When True the output is a numpy. This is a member variable for children classes
                                to modify the output to be specific for some applications but it can be used to
                                for our pipeline when needed, for instance when evaluating the model.
        :param excluded_labels: Labels to be ignored when getting all the labels from bounding boxes or masks
        :param multilabel: If an index in the data is passed, then that position of the data is used for the
                            ordering. It will work with multidata
        """

        super().__init__()
        'Initialization'
        self.base = 'pytorch'
        self.name = name
        self.dim = dim
        self.batch_size = batch_size if batch_size > 0 else 1
        self.use_batch = False if batch_size <= 0 else True
        self.list_data = list_data
        self.n_classes = None
        self._shuffle = shuffle
        self.create_data = create_data
        self.output_as_numpy = output_as_numpy
        self.annotations_are_masks = annotations_are_masks
        self.multilabel = multilabel
        self.default_label = default_label
        self.excluded_labels = excluded_labels

        self.num_iter = num_iter
        self.shuffle_folders = shuffle_folders
        self.initial_class_prob = class_prob
        self.class_prob = None
        self.class_keys = None
        self.fix_iterations = fix_iterations
        self.epoch_number = 0

        self.num_glimpses = num_glimpses
        self.glimpse_size = glimpse_size

        self.group_images = group_images
        self.group_probability = group_probability

        self.iterator = 0

        self.self_supervision = self_supervision
        self.increase_y_size_for_loss = increase_y_size_for_loss

        # Storage for iteration. They are created in on_epoch_end
        self.indexes = list(range(len(list_data)))
        self.labels = []
        self.data = {}

        self.preprocessing_function = preprocessing_function
        self.current_files = []

        self.on_epoch_end()

        self.last_image_paths = []
        self.get_data_path = get_data_path if get_data_path is not None else lambda x: x[0]

        self.store_all_used_data = False
        self.all_used_data = []

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle
        if self._shuffle:
            random.shuffle(self.indexes)
        else:
            self.indexes = list(range(len(self.list_data)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.shuffle_folders or self.fix_iterations:
            return self.num_iter
        else:
            return max(min(1, len(self.indexes)), int(np.floor(len(self.indexes) / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        """indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_urls_temp = [self.list_urls[k] for k in indexes]"""
        #

        grouping_stage = 0
        group_labels = []
        group_values = []

        labels = []
        y = []
        y_sup = []
        Xs = []
        i = 0
        self.current_files = []
        self.last_image_paths = []
        while i < self.batch_size:
            if not self.shuffle_folders:
                position = self.indexes[self.iterator]
                label = self.list_data[position][0]
                values = self.list_data[position][1:]
            else:
                if self.class_prob is None:
                    label = secrets.choice(list(self.data.keys()))
                else:
                    pos = bisect_left(self.class_prob, np.random.uniform(0, 1))
                    label = self.class_keys[pos]
                all_files = self.data[label]
                pos = np.random.randint(0, len(all_files), 1)[0]
                values = all_files[pos]  # list_files_temp.append(all_files[pos])
                labels.append(label)

            if (self.group_images > 1 and np.random.rand(1)[0] < self.group_probability) or grouping_stage == 1:
                group_labels.append(label)
                group_values.append(values)
                if len(group_labels) < self.group_images:
                    grouping_stage = 1
                    self.iterator += 1
                    if self.iterator >= len(self.indexes):
                        self.iterator = 0
                    continue
                else:
                    grouping_stage = 2

            if grouping_stage == 2:
                grouping_stage = 0
                values = list(zip(*group_values))
                label = group_labels
                group_labels = []
                group_values = []

            ims, label, self_label = self._data_generation_single(values, label)
            Xs, y, y_sup, i = self._format_data(ims, label, self_label, Xs, y, y_sup, values, i)

        return self._format_output(Xs, y, y_sup)

    def _format_data(self, ims, label, self_label, Xs, y, y_sup, values, index):
        num_glimpses = max(1, self.num_glimpses)
        if ims is not None and ims[0] is not None and len(ims[0].shape) > 2:
            if isinstance(values[0], str):
                self.current_files.append(values[0])
            if index == 0:
                for im in ims:
                    if num_glimpses > 1:
                        Xs.append(np.empty((num_glimpses * self.batch_size, *self.glimpse_size, im.shape[2]), dtype=np.float32))
                    else:
                        Xs.append(np.empty((self.batch_size, *im.shape), dtype=np.float32))

                for label_i in label:
                    yi = np.zeros(self.batch_size, dtype=np.float32)
                    if hasattr(label_i, 'shape'):
                        yi = np.zeros((self.batch_size, *label_i.shape), dtype=np.float32)
                    y.append(yi)

                for self_label_i in self_label:
                    y_sup_i = np.zeros(self.batch_size, dtype=np.float32)
                    if hasattr(self_label_i, 'shape'):
                        y_sup_i = np.zeros((self.batch_size, *self_label_i.shape), dtype=np.float32)
                    y_sup.append(y_sup_i)

            for ii, im in enumerate(ims):
                Xs[ii][index * num_glimpses: (index + 1) * num_glimpses, ...] = im

            for ii, label_i in enumerate(label):
                y[ii][index, ...] = label_i
            if self.self_supervision:
                for ii, self_label_i in enumerate(self_label):
                    y_sup[ii][index, ...] = self_label_i

            index += 1
            self.last_image_paths.append(self.get_data_path(values))
            if self.store_all_used_data:
                self.all_used_data.append(self.get_data_path(values))

        self.iterator += 1
        if self.iterator >= len(self.indexes):
            self.iterator = 0

        return Xs, y, y_sup, index

    def _format_output(self, Xs, y, y_sup):
        for ii, X in enumerate(Xs):
            Xs[ii] = X.transpose([0, 3, 1, 2])
        if self.annotations_are_masks:
            for ii, y_i in enumerate(y):
                y[ii] = y_i.transpose([0, 3, 1, 2])
            if self.self_supervision:
                for ii, y_sup_i in enumerate(y_sup):
                    y_sup[ii] = y_sup_i.transpose([0, 3, 1, 2])
            # y = np.argmax(y, axis=1).astype(int)

        if not self.use_batch:
            for ii, X in enumerate(Xs):
                Xs[ii] = X[0, ...]
            for ii, y_i in enumerate(y):
                y[ii] = y_i[0, ...]
            if self.self_supervision:
                for ii, y_sup_i in enumerate(y_sup):
                    y_sup[ii] = y_sup_i[0, ...]

        if len(y) == 1:
            y = y[0]
        if len(y_sup) == 1:
            y_sup = y_sup[0]
        if len(Xs) == 1:
            X = Xs[0].astype(np.float32)
        else:
            X = [X.astype(np.float32) for X in Xs]

        if isinstance(y, list):
            for ii, yi in enumerate(y):
                if len(yi.shape) == 1 and self.increase_y_size_for_loss:
                    y[ii] = yi[..., None]
        else:
            if len(y.shape) == 1 and self.increase_y_size_for_loss:
                y = y[..., None]

        if isinstance(y_sup, list):
            for ii, y_supi in enumerate(y):
                if len(y_supi.shape) == 1 and self.increase_y_size_for_loss:
                    y_sup[ii] = y_supi[..., None]
        else:
            if len(y_sup.shape) == 1 and self.increase_y_size_for_loss:
                y_sup = y_sup[..., None]

        if self.self_supervision:
            y = [y, y_sup]

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.iterator = 0
        # list_data has a list of lists or tuples where the first value is the label or mask address if segmentation
        # or caption if captioning and the rest of the values are the image and whatever is required. For instance,
        # we could have a list with (label, image_filename, bounding_box).
        if self.shuffle_folders or self.create_data:
            if self.epoch_number == 0:
                self.data = {}
                if self.multilabel is not None:
                    for items in self.list_data:
                        if not isinstance(self.multilabel, (list, tuple)):
                            self.multilabel = [self.multilabel]
                        labels = []
                        for m in self.multilabel:
                            labels.extend(items[m] if isinstance(items[m], (list, tuple)) else [])
                        if not isinstance(labels, (list, tuple)):
                            labels = [labels]
                        if len(labels) == 0:
                            labels = [self.default_label]
                        for label in labels:
                            if label in self.excluded_labels:
                                label = self.default_label
                            self.data.setdefault(label, []).append(items[1:])
                else:
                    for values in self.list_data:
                        self.data.setdefault(values[0], []).append(values[1:])
                self.n_classes = len(self.data)
            if self.initial_class_prob is not None and self.class_prob is None:
                self.class_prob = self.initial_class_prob.copy()
                self.class_prob = {key: value for key, value in self.class_prob.items() if key in self.data}
                if len(self.class_prob) != len(self.data.keys()):
                    raise ValueError('The length of folder_prob must be the same as the number of labels')
                else:
                    if isinstance(self.class_prob, dict):
                        self.class_keys = list(self.class_prob.keys())
                        values = list(self.class_prob.values())
                        self.class_prob = np.cumsum(values) / np.sum(values)
                    else:
                        self.class_keys = list(self.data.keys())
                        self.class_prob = np.cumsum(self.class_prob) / np.sum(self.class_prob)

        if not self.shuffle_folders and self.shuffle:
            # Indexes are a list from 0 to the number of elements in self.data. Although, self.data could be
            # shuffle it is much faster to randomise a list of numbers. The idea is that self.data could
            # potentially have the image itself and not the address.
            random.shuffle(self.indexes)

        self.epoch_number += 1

    def get_num_glimpse_axis(self, alpha, beta):
        """
        Get the number of glimpse per axis that maintains the
        num_x * glimpse_size_x / size_image_x ~= num_y * glimpse_size_y / size_image_y
        num_x *alpha ~= num_y * beta
        :param alpha: Ratio glimpse_size_x and image_size_x
        :param beta: Ratio glimpse_size_y and image_size_y
        :return: A list with the number of glimpse across x and y
        """
        loss = lambda x: (alpha * x[0] - beta * x[1]) ** 2

        positions = [[1, self.num_glimpses]]
        losses = [loss(positions[0])]

        for i in range(2, self.num_glimpses):
            for ii in range(2, self.num_glimpses):
                if i * ii == self.num_glimpses:
                    positions.append([i, ii])
                    losses.append(loss(positions[-1]))

        return positions[np.argmin(losses)]

    def get_glimpses(self, image):
        """
        Get the glimpses from one image
        :return:
        """
        dim = self.dim
        im = image.astype(float) / np.max(image)

        im_shape = im.shape  # Get shape
        im_ratio = np.float(im_shape[0]) / im_shape[1]  # height/width
        if im_ratio * dim[1] / dim[0] < 1:  # If image is too high
            # Resize to width specified in @dims and keep aspect ratio (up to 1 pixel)
            im = resize(im, (dim[0], np.round(dim[0] / im_ratio) // 2 * 2))
        elif im_ratio * dim[1] / dim[0] > 1:  # If image is too short padd the height with noise
            im = resize(im, (np.round(dim[1] * im_ratio) // 2 * 2, dim[1]))
        else:
            im = resize(im, (dim[0], dim[1]))

        im_shape = im.shape  # Get shape
        alpha = self.glimpse_size[0] / im_shape[0]
        beta = self.glimpse_size[1] / im_shape[1]

        num_glimpses_axis = self.get_num_glimpse_axis(alpha, beta)
        stride = [int(np.round((im_shape[i] - self.glimpse_size[i]) / n_g)) for i, n_g in
                  enumerate(num_glimpses_axis)]

        start_index = [0, 0]
        end_index = [0, 0]
        glimpses = []
        for i in range(num_glimpses_axis[0]):
            start_index[0] = i * stride[0]
            end_index[0] = start_index[0] + self.glimpse_size[0]
            if i == num_glimpses_axis[0] - 1:
                end_index[0] = im_shape[0]
                start_index[0] = end_index[0] - self.glimpse_size[0]
            for ii in range(num_glimpses_axis[1]):
                start_index[1] = ii * stride[1]
                end_index[1] = start_index[1] + self.glimpse_size[1]
                if ii == num_glimpses_axis[1] - 1:
                    end_index[1] = im_shape[1]
                    start_index[1] = end_index[1] - self.glimpse_size[1]

                glimpses.append(im[np.newaxis, start_index[0]: end_index[0], start_index[1]: end_index[1], :])

        return np.concatenate(glimpses, axis=0)

    def normalise_bb(self, value, max_size):
        if value < 0:
            value = max_size
        if value < 1:
            value = int(value * max_size)
        return np.clip(value, 0, max_size)

    def _preprocess_image(self, values, label):
        """
        Read an image and pre-process it. Make sure that the image, label and the self supervision output
        are al lists. In general, they should be lists of numbers of lists of numpy arrays. The reason is that
        we will create as many numpy arrays as elements these lists.
        :param values: All the data passed from the user. It should at least contain 1 value, being an image or a
                        file.
        :return: The image with the correct size
        """
        self_labels = [None]

        # preprocessing_function needs to have as input the same format as the list_labels.
        output = self.preprocessing_function(label, *values, generator_name=self.name)

        if output is None:
            return None, None, None

        if isinstance(output, tuple):
            output = list(output)

        if self.num_glimpses > 1:
            return self.get_glimpses(output[0]), label, self_labels

        if not isinstance(output, list) or (isinstance(output, list) and len(output) == 1):
            if not isinstance(label, list):
                label = [label]
            if not isinstance(output, list):
                output = [output]
            output = [output, label, self_labels]
        else:
            for i, out in enumerate(output):
                if not isinstance(out, (list, tuple)):
                    output[i] = [out]

            if len(output) == 2:
                if self.self_supervision:
                    output = [output[0], label, output[1]]
                else:
                    output.append(self_labels)

        return output

    def _data_generation_single(self, values, label):
        # import matplotlib.pyplot as plt
        #'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        try:
            im, y, y2 = self._preprocess_image(values, label)
        except Exception as e:
            # raise
            return None, None, None

        if im is not None:
            return im, y, y2

        return None, None, None



def collate_fn(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return np.stack(img, 0), np.concatenate(label, 0), path, shapes


class Yolov7Generator(DataGenerator):
    DEFAULT_LOADER_CONFIGURATION = {'collate_fn': collate_fn}

    def __init__(self, *args, increase_y_size_for_loss=False, output_as_numpy=False,  **kwargs):
        kwargs['increase_y_size_for_loss'] = increase_y_size_for_loss
        kwargs['self_supervision'] = True
        kwargs['output_as_numpy'] = output_as_numpy
        super().__init__(*args, **kwargs)
        self.base = 'pytorch'

        # Force batched_images to be True since we are using a generator.
        pf = self.preprocessing_function
        self.preprocessing_function = partial(pf, batched_images=True)

        self.labels = []
        for datum in self.list_data:
            if len(datum[2]) > 0:
                self.labels.append(np.hstack([np.array(datum[3]).reshape([-1, 1]), np.array(datum[2])]))
            else:
                self.labels.append(np.zeros((0, 5)))
        self.indices = self.indexes
        self.n = len(self.indexes)
        self.shapes = None  # In future, this must have the real dimension of the images w, h as ndarray(num_images, 2)

    def _format_data(self, ims, label, current_shapes, Xs, y, shapes, values, index):
        num_glimpses = max(1, self.num_glimpses)
        if ims is not None and ims[0] is not None:  # and len(ims[0].shape) > 2:
            if isinstance(values[0], str):
                self.current_files.append(values[0])
            if index == 0:
                for im in ims:
                    if num_glimpses > 1:
                        Xs.append(np.empty((num_glimpses * self.batch_size, *self.glimpse_size, im.shape[2]),
                                           dtype=im.dtype))
                    else:
                        Xs.append(np.empty((self.batch_size, *im.shape), dtype=im.dtype))

                for label_i in label:
                    yi = np.zeros(self.batch_size, dtype=np.float32)
                    if hasattr(label_i, 'shape'):
                        yi = np.zeros((self.batch_size, *label_i.shape), dtype=np.float32)
                    y.append(yi)

                for current_shapes_i in current_shapes:
                    shapes.append([current_shapes_i])

            for ii, im in enumerate(ims):
                Xs[ii][index * num_glimpses: (index + 1) * num_glimpses, ...] = im

            for ii, label_i in enumerate(label):
                y[ii][index, ...] = label_i
            if self.self_supervision:
                for ii, current_shapes_i in enumerate(current_shapes):
                    shapes[ii].append(current_shapes_i)

            index += 1
            self.last_image_paths.append(self.get_data_path(values))
            if self.store_all_used_data:
                self.all_used_data.append(self.get_data_path(values))

        self.iterator += 1
        if self.iterator >= len(self.indexes):
            self.iterator = 0

        return Xs, y, shapes, index

    def _format_output(self, Xs, y, shapes):
        #if self.use_batch:
        #for ii, X in enumerate(Xs):
        #    Xs[ii] = X.transpose([0, 3, 1, 2])
            #   y = np.argmax(y, axis=1).astype(int)

        if not self.use_batch:
            for ii, X in enumerate(Xs):
                Xs[ii] = X[0, ...]
            for ii, y_i in enumerate(y):
                y[ii] = y_i[0, ...]
            if self.self_supervision:
                for ii, shapes_i in enumerate(shapes):
                    shapes[ii] = shapes_i[0]

        if len(y) == 1:
            y = y[0]
        if len(shapes) == 1:
            shapes = shapes[0]

        if Xs[0].dtype == np.float64:
            Xs = [X.astype(np.float32) for X in Xs]

        if len(Xs) == 1:
            X = Xs[0]

        if isinstance(y, list):
            for ii, yi in enumerate(y):
                if len(yi.shape) == 1 and self.increase_y_size_for_loss:
                    y[ii] = yi[..., None]
        else:
            if len(y.shape) == 1 and self.increase_y_size_for_loss:
                y = y[..., None]

        image_paths = self.last_image_paths[0]
        if isinstance(image_paths, (tuple, list)):
            image_paths = image_paths[0]
        '''if not self.use_batch:
            image_paths = image_paths[0]'''

        if not self.output_as_numpy:
            if isinstance(y, list):
                for ii, yi in enumerate(y):
                    y[ii] = torch.from_numpy(yi)
            else:
                y = torch.from_numpy(y)

            if isinstance(X, list):
                for ii, Xi in enumerate(X):
                    Xi = np.ascontiguousarray(Xi)
                    X[ii] = torch.from_numpy(Xi)
            else:
                X = np.ascontiguousarray(X)
                X = torch.from_numpy(X)

        return X, y, image_paths, shapes