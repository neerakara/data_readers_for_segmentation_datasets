# =========================================================
# =========================================================
import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dialated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         cv_fold_number):
    
    if cv_fold_number == 1:        
        if patient_id % 5 == 0: return 'test'
        elif patient_id % 4 == 0: return 'validation'
        else: return 'train'
        
def get_file_list(input_folder,
                  cv_fold_num):
    
    file_list = {'test': [], 'train': [], 'validation': []}
    
    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            train_test = test_train_val_split(patient_id = int(folder[-3:]), cv_fold_number = cv_fold_num)        
    
            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??_n4.nii.gz')):
                file_list[train_test].append(file)
    
    return file_list
        
def load_without_size_preprocessing(input_folder,
                                    cv_fold_num,
                                    train_test,
                                    idx):
    
    file_list = get_file_list(input_folder,
                              cv_fold_num)
    
    image_file = file_list[train_test][idx]
    
    # ============
    # read image and normalize it to be between 0 and 1
    # ============
    image_dat = utils.load_nii(image_file)
    image = image_dat[0].copy()
    image = utils.normalise_image(image, norm_type='div_by_max')
    
    # ============
    # read label and set RV label to 1, others to 0
    # ============
    label_file = image_file.split('_n4.nii.gz')[0] + '_gt.nii.gz'
    label_dat = utils.load_nii(label_file)
    label = label_dat[0].copy()
    label[label!=1] = 0
        
    return image, label
        
# ====================================================
# Main function that reads the original images and labels and pre-processes them
# ====================================================
def prepare_data(input_folder,
                 output_file,
                 mode,
                 size, # for 3d: (nz, nx, ny), for 2d: (nx, ny)
                 target_resolution, # for 3d: (px, py, pz), for 2d: (px, py)
                 cv_fold_num): 
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    # ============
    # create an empty hdf5 file
    # ============
    hdf5_file = h5py.File(output_file, "w")

    # ============
    # create empty lists for filling header info
    # ============
    diag_list = {'test': [], 'train': [], 'validation': []}
    height_list = {'test': [], 'train': [], 'validation': []}
    weight_list = {'test': [], 'train': [], 'validation': []}
    patient_id_list = {'test': [], 'train': [], 'validation': []}
    cardiac_phase_list = {'test': [], 'train': [], 'validation': []}
    nx_list = {'test': [], 'train': [], 'validation': []}
    ny_list = {'test': [], 'train': [], 'validation': []}
    nz_list = {'test': [], 'train': [], 'validation': []}
    px_list = {'test': [], 'train': [], 'validation': []}
    py_list = {'test': [], 'train': [], 'validation': []}
    pz_list = {'test': [], 'train': [], 'validation': []}

    file_list = {'test': [], 'train': [], 'validation': []}
    num_slices = {'test': 0, 'train': 0, 'validation': 0}

    # ============
    # go through all images and get header info.
    # one round of parsing is done just to get all the header info. The size info is used to create empty fields for the images and labels, with the required sizes.
    # Then, another round of reading the images and labels is done, which are pre-processed and written into the hdf5 file
    # ============
    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            # ============
            # train_test_validation split
            # ============
            train_test = test_train_val_split(patient_id = int(folder[-3:]),
                                              cv_fold_number = cv_fold_num)
            
            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')

            # ============
            # reading this patient's image and collecting header info
            # ============
            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??_n4.nii.gz')):
                
                # ============
                # list with file paths
                # ============
                file_list[train_test].append(file)
                
                diag_list[train_test].append(diagnosis_dict[infos['Group']])
                weight_list[train_test].append(infos['Weight'])
                height_list[train_test].append(infos['Height'])

                patient_id_list[train_test].append(patient_id)

                systole_frame = int(infos['ES'])
                diastole_frame = int(infos['ED'])

                file_base = file.split('.')[0]
                frame = int(file_base.split('frame')[-1][:-3])
                if frame == systole_frame:
                    cardiac_phase_list[train_test].append(1)  # 1 == systole
                elif frame == diastole_frame:
                    cardiac_phase_list[train_test].append(2)  # 2 == diastole
                else:
                    cardiac_phase_list[train_test].append(0)  # 0 means other phase

                nifty_img = nib.load(file)
                nx_list[train_test].append(nifty_img.shape[0])
                ny_list[train_test].append(nifty_img.shape[1])
                nz_list[train_test].append(nifty_img.shape[2])
                num_slices[train_test] += nifty_img.shape[2]
                py_list[train_test].append(nifty_img.header.structarr['pixdim'][2])
                px_list[train_test].append(nifty_img.header.structarr['pixdim'][1])
                pz_list[train_test].append(nifty_img.header.structarr['pixdim'][3])

    # ============
    # writing the small datasets
    # ============
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))

    # ============
    # setting sizes according to 2d or 3d
    # ============
    if mode == '3D': # size [num_patients, nz, nx, ny]
        nz_max, nx, ny = size
        n_train = len(file_list['train']) # number of patients
        n_test = len(file_list['test'])
        n_val = len(file_list['validation'])

    elif mode == '2D': # size [num_z_slices_across_all_patients, nx, ny]
        nx, ny = size
        n_test = num_slices['test']
        n_train = num_slices['train']
        n_val = num_slices['validation']

    else:
        raise AssertionError('Wrong mode setting. This should never happen.')

    # ============
    # creating datasets for images and labels
    # ============
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['labels_%s' % tt] = hdf5_file.create_dataset("labels_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    image_list = {'test': [], 'train': [], 'validation': []}
    label_list = {'test': [], 'train': [], 'validation': []}    

    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0
        patient_counter = 0
        
        for image_file in file_list[train_test]:

            patient_counter += 1

            logging.info('============================================')
            logging.info('Doing: %s' % image_file)

            # ============
            # read image
            # ============
            image_dat = utils.load_nii(image_file)
            image = image_dat[0].copy()
            
            # ============
            # normalize the image to be between 0 and 1
            # ============
            image = utils.normalise_image(image, norm_type='div_by_max')
            
            # ============
            # read label
            # ============
            file_base = image_file.split('_n4.nii.gz')[0]
            label_file = file_base + '_gt.nii.gz'
            label_dat = utils.load_nii(label_file)
            label = label_dat[0].copy()
            
            # ============
            # set RV label to 1 and other labels to 0, as the RVSC dataset only has labels for the RV
            # original labels: 0 bachground, 1 right ventricle, 2 myocardium, 3 left ventricle
            # ============
            label[label!=1] = 0

            # ============
            # original pixel size (px, py, pz)
            # ============
            pixel_size = (image_dat[2].structarr['pixdim'][1],
                          image_dat[2].structarr['pixdim'][2],
                          image_dat[2].structarr['pixdim'][3])

            # ========================================================================
            # PROCESSING LOOP FOR 3D DATA 
            # ========================================================================
            if mode == '3D':
                
                # rescaling ratio
                scale_vector = [pixel_size[0] / target_resolution[0],
                                pixel_size[1] / target_resolution[1],
                                pixel_size[2]/ target_resolution[2]]

                # ==============================
                # rescale image and label
                # ==============================
                image_scaled = transform.rescale(image, scale_vector, order=1, preserve_range=True, multichannel=False, mode='constant')
                label_scaled = transform.rescale(label, scale_vector, order=0, preserve_range=True, multichannel=False, mode='constant')
                
                # ==============================
                # ==============================
                image_scaled = utils.crop_or_pad_volume_to_size_along_z(image_scaled, nz_max)
                label_scaled = utils.crop_or_pad_volume_to_size_along_z(label_scaled, nz_max)

                # ==============================
                # nz_max is the z-dimension provided in the 'size' parameter
                # ==============================
                image_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                label_vol = np.zeros((nx, ny, nz_max), dtype=np.uint8)

                # ===============================
                # going through each z slice
                # ===============================
                for zz in range(nz_max):

                    image_slice = image_scaled[:,:,zz]
                    label_slice = label_scaled[:,:,zz]

                    # cropping / padding with zeros the x-y slice at this z location
                    image_slice_cropped = utils.crop_or_pad_slice_to_size(image_slice, nx, ny)
                    label_slice_cropped = utils.crop_or_pad_slice_to_size(label_slice, nx, ny)

                    image_vol[:,:,zz] = image_slice_cropped
                    label_vol[:,:,zz] = label_slice_cropped


                # ===============================
                # swap axes to maintain consistent orientation as compared to 2d pre-processing
                # ===============================
                image_vol = image_vol.swapaxes(0, 2).swapaxes(1, 2)
                label_vol = label_vol.swapaxes(0, 2).swapaxes(1, 2)
                
                # ===============================
                # append to list that will be written to the hdf5 file
                # ===============================
                image_list[train_test].append(image_vol)
                label_list[train_test].append(label_vol)

                write_buffer += 1

                # ===============================
                # writing the images and labels pre-processed so far to the hdf5 file
                # ===============================
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, image_list, label_list, counter_from, counter_to)
                    _release_tmp_memory(image_list, label_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0

            # ========================================================================
            # PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA 
            # ========================================================================
            elif mode == '2D':

                scale_vector = [pixel_size[0] / target_resolution[0],
                                pixel_size[1] / target_resolution[1]]

                # ===============================
                # go through each z slice, rescale and crop and append.
                # in this process, the z axis will become the zeroth axis
                # ===============================
                for zz in range(image.shape[2]):

                    image_slice = np.squeeze(image[:, :, zz])
                    label_slice = np.squeeze(label[:, :, zz])
                    
                    image_slice_rescaled = transform.rescale(image_slice, scale_vector, order = 1, preserve_range = True, multichannel = False, mode = 'constant')
                    label_slice_rescaled = transform.rescale(label_slice, scale_vector, order = 0, preserve_range = True, multichannel = False, mode = 'constant')

                    image_slice_cropped = utils.crop_or_pad_slice_to_size(image_slice_rescaled, nx, ny)
                    label_slice_cropped = utils.crop_or_pad_slice_to_size(label_slice_rescaled, nx, ny)

                    image_list[train_test].append(image_slice_cropped)
                    label_list[train_test].append(label_slice_cropped)

                    write_buffer += 1

                    # Writing needs to happen inside the loop over the slices
                    if write_buffer >= MAX_WRITE_BUFFER:

                        counter_to = counter_from + write_buffer
                        _write_range_to_hdf5(data, train_test, image_list, label_list, counter_from, counter_to)
                        _release_tmp_memory(image_list, label_list, train_test)

                        # reset stuff for next iteration
                        counter_from = counter_to
                        write_buffer = 0

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, image_list, label_list, counter_from, counter_to)
        _release_tmp_memory(image_list, label_list, train_test)

    # After test train loop:
    hdf5_file.close()

# ====================================================
# 
# ====================================================
def _write_range_to_hdf5(hdf5_data,
                         train_test,
                         img_list,
                         lab_list,
                         counter_from,
                         counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    lab_arr = np.asarray(lab_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['labels_%s' % train_test][counter_from:counter_to, ...] = lab_arr

# ====================================================
# 
# ====================================================
def _release_tmp_memory(img_list,
                        lab_list,
                        train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    lab_list[train_test].clear()
    gc.collect()

# ====================================================
# 
# ====================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=False,
                                cv_fold_num = 1):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'. 2D saves the data slice-by-slice, 3D saves entire volumes
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_size_%s_res_%s_cv_fold_%d.hdf5' % (mode, size_str, res_str, cv_fold_num)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     mode,
                     size,
                     target_resolution,
                     cv_fold_num)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    # a h5py.File object is returned. This acts like a python dictionary.
    # list(data_acdc.keys()) gives the keys in this dictionary.
    return h5py.File(data_file_path, 'r')