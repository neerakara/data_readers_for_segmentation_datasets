# ==================================================================
# import 
# ==================================================================
import load_datasets

# ================================
# set anatomy here.
# ================================
# 'brain' / 'cardiac' / 'prostate'
ANATOMY = 'cardiac'

# ================================
# set dataset here.
# ================================
# for brain: 'HCP_T1' / 'HCP_T2' / 'ABIDE_caltech' / 'ABIDE_stanford'
# for cardiac: 'ACDC' / 'RVSC'
# for prostate: 'NCI' / 'PIRAD_ERC'
DATASET = 'RVSC' 

# ================================
# set train / test / validation here
# ================================
# 'train' / 'test' / 'validation'
TRAIN_TEST_VALIDATION = 'validation'

images, labels = load_datasets.load_dataset(anatomy = ANATOMY,
                                            dataset = DATASET,
                                            train_test_validation = TRAIN_TEST_VALIDATION)