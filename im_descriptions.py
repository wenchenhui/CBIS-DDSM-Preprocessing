import glob
import os
import re
from collections import defaultdict, namedtuple

import pandas as pd

try:
    import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return iter(it)

DATASET_PATH = '.'
IM_DESCRIPTIONS_PATH = 'descriptions.csv'

CALC_TEST_DESC_PATH = 'descriptions/calc_case_description_test_set.csv'
CALC_TRAIN_DESC_PATH = 'descriptions/calc_case_description_train_set.csv'
MASS_TEST_DESC_PATH = 'descriptions/mass_case_description_test_set.csv'
MASS_TRAIN_DESC_PATH = 'descriptions/mass_case_description_train_set.csv'


def get_mass_im_paths(dataset_path):
    """Return an iterator over dcm filepaths contained in directories
    whose names start with 'Mass' (so calcification images are
    excluded), and rooted at `dataset_path`.
    """
    return glob.iglob(
        os.path.join(dataset_path, '**', 'Mass*', '**', '*.dcm'),
        recursive=True)


def get_im_root(im_path):
    return os.path.basename(
        # Go up 3 directories:
        os.path.normpath(os.path.join(im_path, '..', '..', '..')))


def get_im_description(im_root):
    # im_root example: Mass-Training_P_00001_LEFT_CC_1
    regex = (r'^(?P<lesion_type>.*?)-(?P<set>.*?)_(?P<patient_id>P_.*?)'
             r'_(?P<direction>.*?)_(?P<view>.*?)(?P<is_overlay>_\d)?$')
    return re.search(regex, im_root).groups()


def is_cropped(im_path):
    """If the file is less than 1 MB it's *most likely* a cropped image.
    Easier than querying the dataframe.
    """
    return os.path.getsize(im_path) < 2 ** 20


_descriptions = {}
def get_df(lesion_type, set_):
    if lesion_type == 'Calc':
        description_path = (CALC_TEST_DESC_PATH if set_ == 'Test'
                            else CALC_TRAIN_DESC_PATH)
    else:
        description_path = (MASS_TEST_DESC_PATH if set_ == 'Test'
                            else MASS_TRAIN_DESC_PATH)
    # Lazy initialization of dataframes:
    if description_path not in _descriptions:
        _descriptions[description_path] = pd.read_csv(description_path)
    return _descriptions[description_path]


def get_pathology(lesion_type, set_, is_overlay, im_root):
    df = get_df(lesion_type, set_)
    col_name = 'ROI mask file path' if is_overlay else 'image file path'
    return (df[[path.startswith(im_root) for path in df[col_name]]]
                .iat[0, df.get_loc('pathology')])


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
    if is_cropped(im_path):
        return
    im_root = get_im_root(im_path)
    lesion_type, set_, patient_id, direction, view, is_overlay = (
        get_im_description(im_root))
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
    for im_path in tqdm(get_mass_im_paths(dataset_path)):
        add_im_description(im_descriptions, im_path)
    return pd.DataFrame(im_descriptions.values())


if __name__ == "__main__":
    get_im_descriptions(DATASET_PATH).to_csv(IM_DESCRIPTIONS_PATH)
