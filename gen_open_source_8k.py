#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import cv2

## Define some configuration variables:
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1  # no. of times to use the same image
SECS_PER_IMG = 5  # max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
LANG_PATH = 'data'
DATA_PATH = 'bkg_images'
IMG_PATH = 'bkg_images/bg_img'

# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results_8k/SynthText_open_source_8k.h5'
OUT_DIR = 'results_8k'


def get_data():
    """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
    if not osp.exists(DB_FNAME):
        try:
            print('\tdownloading data (56 M) from: ' + DATA_URL)
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            print('\n\tdata saved at:' + DB_FNAME)
            sys.stdout.flush()
        except:
            print('Data not found and have problems downloading.')
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')


def add_res_to_db(imgname, res, db):
    """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=np.zeros(1))
        #db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        db['data'][dname].attrs['txt'] = res[i]['txt']
        text_utf8 = [char.encode('utf8') for char in res[i]['txt']]
        db['data'][dname].attrs['txt_utf8'] = text_utf8
        
def save_res_to_imgs(imgname, res):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        filename = "{}/{}_{}.png".format(OUT_DIR, imgname, i)
        # Swap bgr to rgb so we can save into image file
        img = res[i]['img'][..., [2, 1, 0]]
        cv2.imwrite(filename, img)

def main(viz=False):
    depth_db = h5py.File(os.path.join(DATA_PATH, 'depth.h5'), 'r')
    seg_db = h5py.File(os.path.join(DATA_PATH, 'seg.h5'), 'r')
    
    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print('Storing the output in: ' + OUT_FILE)

    # get the names of the image files in the dataset:
    imnames = sorted(depth_db.keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    RV3 = RendererV3(LANG_PATH, max_time=SECS_PER_IMG, lang=args.lang)
    for i in range(start_idx, end_idx):
        imname = imnames[i]
        print (f"Rendering for {imname}")
        
        try:
            # get the image:
            img = Image.open(osp.join(IMG_PATH, imname)).convert('RGB')
            depth = depth_db[imname][:].T
            depth = depth[:,:,0]
            
            # get segmentation:
            seg = seg_db['mask'][imname][:].astype('float32')
            area = seg_db['mask'][imname].attrs['area']
            label = seg_db['mask'][imname].attrs['label']

            # re-size uniformly:
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.Resampling.LANCZOS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.Resampling.NEAREST))

            print('%d of %d' % (i, end_idx - 1))
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, viz=viz)
            save_res_to_imgs(imname, res)
            
            if len(res) > 0:
                # non-empty : successful in placing text:
                add_res_to_db(imname, res, out_db)

            # visualize the output:
            if viz:
                save_res_to_imgs(imname, res)
                if 'q' in input('continue? (enter to continue, q to exit): '):
                    break
        except:
            traceback.print_exc()
            print('>>>> CONTINUING....')
            continue
    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=False,
                        help='flag for turning on visualizations')
    parser.add_argument('--lang', default='ENG',
                        help='Select language : ENG or JPN')
    args = parser.parse_args()
    main(args.viz)
