import fmd

from fmd.ds300w import DS300W
from fmd.ds300vw import DS300VW
from fmd.afw import AFW
from fmd.helen import HELEN
from fmd.ibug import IBUG
from fmd.lfpw import LFPW
from fmd.wflw import WFLW

from fmd.mark_dataset.util import draw_marks

from augment import (flip_randomly, generate_heatmaps, normalize,
                     rotate_randomly, scale_randomly)

from util_dataset import *

import cv2
import numpy as np
import os
import shutil
from copy import deepcopy

des = '/home/lap14880/hieunmt/facepose/facepose_gendata/dataset'

try:
    shutil.rmtree(des)
except:
    pass

des_img = des + '/images'
des_mark = des + '/marks'
os.mkdir(des)
os.mkdir(des_img)
os.mkdir(des_mark)

def get_images_marks(im_size, rot_angle=30, make_augment=False):
    img_size = (im_size, im_size)

    # Set the dataset directory you are going to use.
    from_route = '/home/lap14880/hieunmt/facepose/faceposelandmark'
    ds300w_dir = f"{from_route}/download/300W"
    ds300vw_dir = f"{from_route}/download/300VW"
    afw_dir = f"{from_route}/download/afw"
    helen_dir = f"{from_route}/download/helen"
    ibug_dir = f"{from_route}/download/ibug"
    lfpw_dir = f"{from_route}/download/LFPW"
    wflw_dir = f"{from_route}/download/WFLW"
    aflw2000_3d_dir = f"{from_route}/download/AFLW2000"

    # Construct the datasets.

    # # 300W
    ds_300w = fmd.ds300w.DS300W("300w")
    ds_300w.populate_dataset(ds300w_dir)

    # 300VW
    ds_300vw = fmd.ds300vw.DS300VW("300vw")
    ds_300vw.populate_dataset(ds300vw_dir)

    # AFW
    ds_afw = fmd.afw.AFW("afw")
    ds_afw.populate_dataset(afw_dir)

    # HELEN
    ds_helen = fmd.helen.HELEN("helen")
    ds_helen.populate_dataset(helen_dir)

    # IBUG
    ds_ibug = fmd.ibug.IBUG("ibug")
    ds_ibug.populate_dataset(ibug_dir)

    # LFPW
    ds_lfpw = fmd.lfpw.LFPW("lfpw")
    ds_lfpw.populate_dataset(lfpw_dir)

    # WFLW
    ds_wflw = fmd.wflw.WFLW(True, "wflw")
    ds_wflw.populate_dataset(wflw_dir)

    # AFLW2000-3D
    ds_aflw2k3d = fmd.AFLW2000_3D("AFLW2000_3D")
    ds_aflw2k3d.populate_dataset(aflw2000_3d_dir)

    # datasets = [ds_300vw, ds_300w, ds_aflw2k3d,
    #             ds_afw, ds_helen, ds_ibug, ds_lfpw, ds_wflw]

    datasets = [ds_300vw, ds_300w, ds_aflw2k3d, ds_afw, ds_helen, ds_ibug, ds_lfpw, ds_wflw]

    # for debug
    # datasets = [ds_afw]

    # How many samples do we have?
    print("Total samples: {}".format(sum(ds.meta["num_samples"] for ds in datasets)))

    mo = MarkOperator()
    cnt = 0

    for ds in datasets:
        print(ds)
        for sample in ds:
            image = sample.read_image()
            # marks = sample.get_key_marks()
            marks = sample.marks

            if ds.meta['name'] == 'wflw':
                marks = list98_to_list68(marks)
                marks = np.array(marks, np.float32)

            image_translated, trans_vector = move_face_to_center(image, marks, mo)
            marks_translated = marks[:, :2] + trans_vector

            # Second, align the face. This happens in the 2D space.
            # image_rotated, degrees = rotate_to_vertical(image_translated, sample, mo)
            img_height, img_width, _ = image.shape
            # marks_rotated = mo.rotate(marks_translated, degrees/180*np.pi, (img_width/2, img_height/2))

            # Third, try to crop the face area out. Pad the image if necessary.

            image_cropped, padding, bbox = crop_face(image_translated, marks, scale=1.7)
            mark_cropped = marks_translated + padding - np.array([bbox[0], bbox[1]])

            # Last, resize the face area. I noticed Google is using 192px.
            image_resized = cv2.resize(image_cropped, img_size)
            mark_resized = mark_cropped * (im_size / image_cropped.shape[0])

            cv2.imwrite(f"{des_img}/{cnt}.jpg", image_resized)
            np.save(f'{des_mark}/{cnt}.npy', mark_resized)
            cnt += 1

            if make_augment:
                def do_augment(image, marks):
                    marks = np.hstack((marks, np.zeros((marks.shape[0], 1))))

                    image, marks = rotate_randomly(image, marks, (-rot_angle, rot_angle))

                    # Scale the image randomly.
                    image, marks = scale_randomly(image, marks, output_size=(im_size, im_size))

                    # Flip the image randomly.
                    image, marks = flip_randomly(image, marks) 

                    marks = marks[..., :-1]

                    return image, marks

                image, marks = do_augment(deepcopy(image_resized), deepcopy(mark_resized))
                cv2.imwrite(f"{des_img}/{cnt}.jpg", image)
                np.save(f'{des_mark}/{cnt}.npy', marks)
                cnt += 1

                image, marks = do_augment(deepcopy(image_resized), deepcopy(mark_resized))
                cv2.imwrite(f"{des_img}/{cnt}.jpg", image)
                np.save(f'{des_mark}/{cnt}.npy', marks)
                cnt += 1

        print('cnt', cnt)

if __name__ == '__main__':
    get_images_marks(im_size=192, 
                     rot_angle=30,
                     make_augment=True)

    # Load the numpy files
    def load_npy(npy_path):
        feature = np.load(npy_path)
        return feature

    name = '1'
    data_path = '/home/lap14880/hieunmt/facepose/facepose_gendata/dataset'
    img_path = data_path + '/images/' + name + '.jpg'
    mark_path = data_path + '/marks/' + name + '.npy'

    img = cv2.imread(img_path)
    mark = load_npy(mark_path)
    cv2.imwrite(f"/home/lap14880/hieunmt/facepose/facepose_gendata/datatool/sample_inp.jpg", img)
    draw_marks(img, mark)
    cv2.imwrite(f"/home/lap14880/hieunmt/facepose/facepose_gendata/datatool/sample_mark.jpg", img)
