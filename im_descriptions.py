import glob
import os
import re
from collections import defaultdict, namedtuple
from functools import lru_cache

import pandas as pd
from tqdm import tqdm

DATASET_PATH = '.'
DESCRIPTIONS_PATHS = {
    ('Calc', 'Test'): 'descriptions/calc_case_description_test_set.csv',
    ('Calc', 'Training'): 'descriptions/calc_case_description_train_set.csv',
    ('Mass', 'Test'): 'descriptions/mass_case_description_test_set.csv',
    ('Mass', 'Training'): 'descriptions/mass_case_description_train_set.csv',
}
IM_DESCRIPTIONS_PATH = 'descriptions.csv'


def get_im_paths(dataset_path, substr=None):
    """Return an iterator over dcm filepaths contained in directories whose
    names contain `substr` (so if `substr='Calc'` then only calcification
    images are included), and are rooted at `dataset_path`.
    """
    dir_pattern = ('**', f'*{substr}*', '**') if substr else ('**',)
    return glob.iglob(
        os.path.join(dataset_path, *dir_pattern, '*.dcm'),
        recursive=True)


def is_cropped2(im_path):
    return os.path.getsize(im_path) < 5 * 2 ** 20  # B = 5 MB


def get_im_root(im_path):
    return os.path.basename(
        # Go up 3 directories:
        os.path.normpath(os.path.join(im_path, '..', '..', '..')))


def get_im_description(im_root):
    # im_root example: Mass-Training_P_00001_LEFT_CC_1
    regex = (r'^(?P<lesion_type>.*?)-(?P<set>.*?)_(?P<patient_id>P_.*?)'
             r'_(?P<direction>.*?)_(?P<view>.*?)(?P<is_overlay>_\d)?$')
    return re.search(regex, im_root).groups()


@lru_cache(maxsize=4)
def get_df(lesion_type, set_):
    return pd.read_csv(DESCRIPTIONS_PATHS[(lesion_type, set_)])


def is_cropped(lesion_type, set_, im_root, im_name):
    df = get_df(lesion_type, set_)
    cropped_path = next(path for path in df['cropped image file path']
                        if path.startswith(im_root))
    # rstrip() to remove newlines:
    return os.path.basename(cropped_path.rstrip()) == im_name


def get_pathology(lesion_type, set_, is_overlay, im_root):
    df = get_df(lesion_type, set_)
    col_name = 'ROI mask file path' if is_overlay else 'image file path'
    i_row = next(i for (i, path) in df[col_name].items()
                 if path.startswith(im_root))
    return df.at[i_row, 'pathology']


def add_im_description(im_descriptions, im_path):
    """ImDescription: Dict[
        lesion_type,
        set,
        patient_id,
        direction,
        view,
        pathology,
        path,
        mask_path,
    ]
    """
    if is_cropped2(im_path):
        return
    im_root = get_im_root(im_path)
    lesion_type, set_, patient_id, direction, view, is_overlay = (
        get_im_description(im_root))
    """if is_overlay and is_cropped(lesion_type, set_, im_root,
                                 os.path.basename(im_path)):
        return
    """
    im_key = (patient_id, direction, view)
    im_type = 'mask_path' if is_overlay else 'path'
    if im_key in im_descriptions:
        im_descriptions[im_key][im_type] = im_path
    else:
        im_descriptions[im_key] = {
            'lesion_type': lesion_type,
            'set': set_,
            'patient_id': patient_id,
            'direction': direction,
            'view': view,
            'pathology': get_pathology(lesion_type, set_, is_overlay, im_root),
            im_type: im_path
        }


def get_im_descriptions(dataset_path):
    im_descriptions = {}
    for im_path in tqdm(get_im_paths(dataset_path, 'Mass')):
        add_im_description(im_descriptions, im_path)
    return pd.DataFrame(im_descriptions.values())


if __name__ == "__main__":
    get_im_descriptions(DATASET_PATH).to_csv(IM_DESCRIPTIONS_PATH)
