import random

import numpy
import torch
import tqdm
from PIL import Image
from scipy.cluster.vq import kmeans


def set_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_label(filenames):
    x = {}
    for filename in filenames:
        try:
            image = Image.open(filename)
            image.verify()
            shape = image.size
            assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
            with open(filename.replace('images', 'labels').replace('jpg', 'txt'), 'r') as f:
                label = numpy.array([x.split() for x in f.read().splitlines()], dtype=numpy.float32)
            if len(label) == 0:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            x[filename] = [label, shape]
        except FileNotFoundError:
            pass
    return x


def generate_anchors(labels, shapes, n=9, input_size=640, anchor_t=4.0, gen=1000):
    anchor_t = 1. / anchor_t

    def fitness(y):  # mutation fitness
        y = y.astype('float32')
        r = wh[:, None] / y[None]
        y = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = y.max(1)[0]
        return (best * (best > anchor_t).float()).mean()  # fitness

    # Get label wh
    shapes = input_size * shapes / shapes.max(1, keepdims=True)
    wh = numpy.concatenate([l[:, 3:5] * s for s, l in zip(shapes, labels)])  # wh

    # Filter
    wh = wh[(wh >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)

    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm.tqdm(range(gen))  # progress bar
    for _ in pbar:
        v = numpy.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = (numpy.random.random(sh) < mp) * numpy.random.random() * numpy.random.randn(*sh) * s + 1
            v = v.clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f

    k = k[numpy.argsort(k.prod(1))]  # sort small to large
    for x in k:
        print(round(x[0]), round(x[1]))


def main():
    filenames = []
    with open('../Dataset/COCO/train.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../Dataset/COCO/images/train2017/' + filename)
    data = load_label(filenames)
    labels, shapes = zip(*data.values())

    labels = list(labels)
    shapes = numpy.array(shapes, dtype=numpy.float64)
    generate_anchors(labels, shapes)


if __name__ == '__main__':
    set_seed()
    main()
