import os

import numpy
import matplotlib.pyplot as plt
from PIL import Image


def __clean_or_make_dir(output_dir):
    if os.path.exists(output_dir):
        def del_file(path):
            dir_list = os.listdir(path)
            for item in dir_list:
                item = os.path.join(path, item)
                del_file(item) if os.path.isdir(item) else os.remove(item)
        try:
            del_file(output_dir)
        except Exception as e:
            print(e)
            print('please remove the files of output dir.')
            exit(-1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def draw_histogram(count_dict: dict, x_label='X', y_label='Y', max_y=None, title=None):
    labels = []
    counts = []
    for item in count_dict.items():
        labels.append(item[0])
        counts.append(item[1])
    max_y_lim = max(counts) if max_y is None else max_y
    plt.figure()
    plt.bar(labels, counts, width=0.7, align='center')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, max_y_lim + 10)
    plt.show()


def draw_array_img(array, output_dir, normalizing=False):
    """
    draw images of array
    :param normalizing: whether to normalize.
    :param output_dir: directory of images.
    :param array: 3-D array [batch, width, height]
    """
    assert numpy.array(array).ndim == 3
    __clean_or_make_dir(output_dir)

    def array2Picture(arr, name):
        img = Image.fromarray(arr * 255)
        img = img.convert('L')
        img.save(os.path.join(output_dir, "img-%d.jpg" % name))

    def normalize(arr):
        min_v = numpy.min(arr)
        max_v = numpy.max(arr)
        det_v = max_v - min_v
        for i, row in enumerate(arr):
            for j, x in enumerate(row):
                arr[i][j] = (x - min_v) / det_v
        return arr

    for index, mtx in enumerate(array):
        mtx = normalize(mtx) if normalizing else mtx
        array2Picture(numpy.array(mtx), index)


def draw_matrix(matrix, title=''):

    def draw(m, t):
        height = m.shape[0]
        width = m.shape[1]
        plt.figure(figsize=(12, 8))
        plt.title(t)
        plt.pcolormesh(m)
        plt.xlabel('Dimension Second')
        plt.xlim((0, width))
        plt.ylim((height, 0))
        plt.ylabel('Dimension First')
        plt.colorbar()
        plt.show()

    n_dims = numpy.array(matrix).ndim
    if n_dims == 2:
        draw(matrix, title)
    elif n_dims == 3:
        for i, mat in enumerate(matrix):
            draw(mat, title + ' ' + str(i))
    else:
        raise ValueError(
            'Unsupported dimensions %d' % n_dims)


def draw_line_chart(steps, train_acc, eval_acc):
    plt.plot(steps, train_acc, steps, eval_acc)
    plt.show()


def draw_distribution(tensor, T=80, title=None):
    """
     Draw the distribution of a Tensor.
    :param title:
    :param T: number of slices.
    :param tensor: 2-D or 3-D tensor.
    """
    if len(numpy.array(tensor).shape) == 3:
        tensor = numpy.mean(tensor, axis=0, keepdims=False)
    elem_max = numpy.max(tensor)
    elem_min = numpy.min(tensor)
    elem_wid = (elem_max - elem_min) / T
    elem_mean = numpy.mean(tensor)
    elem_dict = {}
    for row in tensor:
        for elem in row:
            key = int((elem - elem_mean) / elem_wid)
            if key == 0 and elem > elem_mean:
                continue
            elem_dict[key] = elem_dict.get(key, 0) + 1
    draw_histogram(elem_dict, title=title)
