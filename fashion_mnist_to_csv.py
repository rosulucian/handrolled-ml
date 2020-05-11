import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.getenv('MNIST_FASHION_PATH')


def filesize(f):
    pos = f.tell()
    f.seek(0, 2)
    size = f.tell()
    f.seek(pos, 0)
    return size


def convert(imgfile, labelfile, outfile='out.csv'):
    outfile = os.path.join(os.path.dirname(img_file), outfile)
    pix_size = 28 * 28

    images = []

    with open(imgfile, 'rb') as img:
        with open(labelfile, 'rb') as lbl:
            img_offset = 16
            lbl_offset = 8

            img_size = filesize(img) - img_offset
            lbl_size = filesize(lbl) - lbl_offset
            assert(img_size / pix_size == lbl_size)

            img.read(img_offset)
            lbl.read(lbl_offset)

            # TODO: use buffers
            for i in range(lbl_size):
                image = [ord(lbl.read(1))]
                for j in range(pix_size):
                    image.append(ord(img.read(1)))

                images.append(image)

    with open(outfile, 'w') as out:
        out.write("label,")
        out.write(",".join(f'pixel{pix}' for pix in range(pix_size))+"\n")

        for img in images:
            out.write(",".join(str(pix) for pix in img)+"\n")


#  train
img_file = os.path.join(data_path, 'train-images-idx3-ubyte')
label_file = os.path.join(data_path, 'train-images-idx1-ubyte')
convert(img_file, label_file, 'train_fashion_mnist.csv')


#  test
img_file = os.path.join(data_path, 't10k-images-idx3-ubyte')
label_file = os.path.join(data_path, 't10k-labels-idx1-ubyte')
convert(img_file, label_file, 'test_fashion_mnist.csv')
