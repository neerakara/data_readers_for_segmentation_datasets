# ==================================================================
# import 
# ==================================================================
import load_datasets

# ================================
# set anatomy here
# ================================
# 'brain' / 'cardiac' / 'prostate'
# ================================
ANATOMY = 'prostate'

# ================================
# set dataset here.
# ================================
# for brain: 'HCP_T1' / 'HCP_T2' / 'ABIDE_caltech' / 'ABIDE_stanford'
# for cardiac: 'ACDC' / 'RVSC'
# for prostate: 'NCI' / 'PIRAD_ERC'
# ================================
DATASET = 'NCI' 

# ================================
# set train / test / validation here
# ================================
# 'train' / 'test' / 'validation'
TRAIN_TEST_VALIDATION = 'train'

# ================================
# read images and segmentation labels
# ================================
images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
                                            dataset = DATASET,
                                            train_test_validation = TRAIN_TEST_VALIDATION,
                                            first_run = False)  # <-- SET TO TRUE FOR THE FIRST RUN (enables preliminary preprocessing e.g. bias field correction)
