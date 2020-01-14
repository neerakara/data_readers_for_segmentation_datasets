# =========================================================
# =========================================================
import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import utils
import pydicom as dicom
import subprocess
from skimage.draw import polygon

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         cv_fold_number):
    
    if cv_fold_number == 1:
        if patient_id % 4 == 0: return 'test'
        elif patient_id % 3 == 0: return 'validation'
        else: return 'train'
        
# ===============================================================
# ===============================================================
def get_patient_folders(input_folder,
                        cv_fold_number):

    image_folder_list = {'train': [], 'test': [], 'validation': []}
    label_folder_list = {'train': [], 'test': [], 'validation': []}
    patient_ids_list = {'train': [], 'test': [], 'validation': []}

    for folder in os.listdir(input_folder):   
        
        patient_id = int(folder[-2:])        
        
        train_test = test_train_val_split(patient_id, cv_fold_number)
        patient_ids_list[train_test].append(patient_id)
        
        patient_folder = os.path.join(input_folder, folder)
        
        for inner_folder in os.listdir(patient_folder):
            
            folder_path = os.path.join(patient_folder, inner_folder)
            
            if 'dicom' in folder_path:
                image_folder_list[train_test].append(folder_path)
                
            elif 'contours' in folder_path:
                label_folder_list[train_test].append(folder_path)

    return image_folder_list, label_folder_list, patient_ids_list

# ====================================================
# ====================================================
def get_image_details(image_folder_path):

    # ================================
    # create a list of all dicom slices filenames
    # ================================
    list_of_dicom_filenames = []
    for dirName, subdirList, fileList in os.walk(image_folder_path):        
        fileList.sort()        
        for filename in fileList:
            list_of_dicom_filenames.append(os.path.join(dirName,filename))
    
    ds = dicom.read_file(list_of_dicom_filenames[0])    
    nx = ds.Rows
    ny = ds.Columns
    nz = len(list_of_dicom_filenames) // 20
    px = ds.PixelSpacing[0]
    py = ds.PixelSpacing[1]
    pz = ds.SliceThickness
            
    return px, py, pz, nx, ny, nz, list_of_dicom_filenames, ds.pixel_array.dtype

# ====================================================
# ====================================================
def read_image(image_folder_path,
               ed_es_diff,
               nifti_available = True):
    
    # ed_es_diff: number of slices between ED and ES
    px, py, pz, nx, ny, nz, list_of_dicom_filenames, pix_dtype = get_image_details(image_folder_path)
    nifti_img_path = image_folder_path[:image_folder_path.rfind('/') + 1]
    
    if nifti_available is False:

        imgDims = (nx, ny, nz)
        img_ED = np.zeros(imgDims, dtype = pix_dtype)
        img_ES = np.zeros(imgDims, dtype = pix_dtype)
    
        # ================================
        # read dicom series and create image volume
        # ================================
        for zz in range(len(list_of_dicom_filenames)):
            
            ds = dicom.read_file(list_of_dicom_filenames[zz])    
            
            slice_id = int(list_of_dicom_filenames[zz][-8:-4])
            
            if slice_id % 20 is 0: # ED
                img_ED[:, :, slice_id // 20] = ds.pixel_array        
            
            elif (slice_id - ed_es_diff) % 20 is 0: # ES
                img_ES[:, :, (slice_id - ed_es_diff) // 20] = ds.pixel_array
            
        # ================================
        # save as nifti, this sets the affine transformation as an identity matrix
        # ================================    
        utils.save_nii(img_path = nifti_img_path + 'img_ED.nii.gz', data = img_ED, affine = np.eye(4))
        utils.save_nii(img_path = nifti_img_path + 'img_ES.nii.gz', data = img_ES, affine = np.eye(4))
    
        # ================================
        # do bias field correction
        # ================================
        for e in ['ED', 'ES']:
            input_img = nifti_img_path + 'img_' + e + '.nii.gz'
            output_img = nifti_img_path + 'img_' + e + '_n4.nii.gz'
            subprocess.call(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", input_img, output_img])
    
    # ================================    
    # read bias corrected image
    # ================================    
    img_ED = utils.load_nii(img_path = nifti_img_path + 'img_ED_n4.nii.gz')[0]
    img_ES = utils.load_nii(img_path = nifti_img_path + 'img_ES_n4.nii.gz')[0]
            
    return img_ED, img_ES, px, py, pz

# ====================================================
# ====================================================
def read_label(label_folder_path,
               shape,
               ed_es_diff,
               nifti_available = True):
    
    # ed_es_diff: number of slices between ED and ES
    
    nifti_lbl_path = label_folder_path[:label_folder_path.rfind('/') + 1]
    
    if nifti_available is False:
    
        label_ED = np.zeros(shape, dtype = np.uint8)
        label_ES = np.zeros(shape, dtype = np.uint8)
    
        for text_file in os.listdir(label_folder_path):
            
            if 'icontour' in text_file:
            
                text_file_path = os.path.join(label_folder_path, text_file)
            
                slice_id = int(text_file[4:8])
    
                slice_labels = open(text_file_path, "r") 
    
                x = []
                y = []
                for l in slice_labels.readlines():
                    row = l.split()                
                    x.append(int(float(row[1])))
                    y.append(int(float(row[0])))                
                slice_labels.close()
                
                # fit a polygon to the points - this provides a way to make a binary segmentation mask
                xx, yy = polygon(x, y)
                
                # set the pixels inside the mask as 1
                if slice_id % 20 is 0: # ED
                    label_ED[xx, yy, slice_id // 20] = 1
                
                elif (slice_id - ed_es_diff) % 20 is 0: # ES
                    label_ES[xx, yy, (slice_id - ed_es_diff) // 20] = 1
                    
        # ================================  
        # save as nifti
        # ================================  
        utils.save_nii(img_path = nifti_lbl_path + 'lbl_ED.nii.gz', data = label_ED, affine = np.eye(4))
        utils.save_nii(img_path = nifti_lbl_path + 'lbl_ES.nii.gz', data = label_ES, affine = np.eye(4))
        
    # ================================    
    # read nifti labels
    # ================================    
    label_ED = utils.load_nii(img_path = nifti_lbl_path + 'lbl_ED.nii.gz')[0]
    label_ES = utils.load_nii(img_path = nifti_lbl_path + 'lbl_ES.nii.gz')[0]

    return label_ED, label_ES        

# ====================================================
# ====================================================            
def load_without_size_preprocessing(input_folder,
                                    cv_fold_num,
                                    train_test,
                                    idx):
    
    image_folders_list, label_folders_list, patient_id_list = get_patient_folders(input_folder, cv_fold_num)    
    
    # ==============================
    # First, get ed_es_diff for this subject
    # ==============================
    for item in os.listdir(image_folders_list[train_test][idx//2][:-8]):
        if 'list.txt' in item:
            text_file = open(image_folders_list[train_test][idx//2][:-8] + item, "r") 
            slice_ids_with_annotations = []
            for l in text_file.readlines():
                slice_ids_with_annotations.append(int(float(l[-25:-21])))
            text_file.close()
            
    for slice_id in np.unique(slice_ids_with_annotations):
        if slice_id % 20 != 0:
            ed_es_diff = int(slice_id % 20)
            continue

    # ==============================
    # read image and label
    # ==============================
    image_ED, image_ES, px, py, pz = read_image(image_folders_list[train_test][idx//2], ed_es_diff)
    label_ED, label_ES = read_label(label_folders_list[train_test][idx//2], image_ED.shape, ed_es_diff)

    img_ED = image_ED.copy()
    img_ES = image_ES.copy()
    lab_ED = label_ED.copy()
    lab_ES = label_ES.copy()

    # ============
    # normalize the image to be between 0 and 1
    # ============
    img_ED = utils.normalise_image(img_ED, norm_type='div_by_max')
    img_ES = utils.normalise_image(img_ES, norm_type='div_by_max')
    
    # ============
    # decide if ES or ED needs to be returned
    # ============
    print((idx%2))
    if int(idx%2) is 0:
        image = img_ED
        label = lab_ED
    else:
        image = img_ES
        label = lab_ES
        
    return image, label

# ====================================================
# 
# ====================================================
def prepare_data(input_folder,
                 output_file,
                 mode,
                 size,
                 target_resolution,
                 cv_fold_num):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    cardiac_phase_list = {'test': [], 'train': [], 'validation': []}
    nx_list = {'test': [], 'train': [], 'validation': []}
    ny_list = {'test': [], 'train': [], 'validation': []}
    nz_list = {'test': [], 'train': [], 'validation': []}
    px_list = {'test': [], 'train': [], 'validation': []}
    py_list = {'test': [], 'train': [], 'validation': []}
    pz_list = {'test': [], 'train': [], 'validation': []}

    num_slices = {'test': 0, 'train': 0, 'validation': 0}

    # ==============================
    # read all images and save header info
    # ==============================
    image_folders_list, label_folders_list, patient_id_list = get_patient_folders(input_folder, cv_fold_num)
    
    for tt in ['train', 'test', 'validation']:
        
        for sub_id in range(len(image_folders_list[tt])):
        
            image_details = get_image_details(image_folders_list[tt][sub_id])

            for _ in range(2): # append details for 2 volumes - ED and ES            

                px_list[tt].append(image_details[0])
                py_list[tt].append(image_details[1])
                pz_list[tt].append(image_details[2])
                
                nx_list[tt].append(image_details[3])
                ny_list[tt].append(image_details[4])
                nz_list[tt].append(image_details[5])
                
                num_slices[tt] += image_details[5]
                
            cardiac_phase_list[tt].append(1) # 'ES'
            cardiac_phase_list[tt].append(2) # 'ED'

    # ==============================
    # Write the small datasets
    # ==============================
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))

    # ==============================
    # set dimensions for the hdf5 file
    # ==============================
    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']

    # ==============================
    # Create datasets for images and labels
    # ==============================
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['labels_%s' % tt] = hdf5_file.create_dataset("labels_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    label_list = {'test': [], 'train': [], 'validation': []}
    image_list = {'test': [], 'train': [], 'validation': []}

    # ==============================
    # read each image and label
    # ==============================
    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0
        patient_counter = 0
        
        for sub_num in range(len(image_folders_list[train_test])):

            patient_counter += 1

            logging.info('============================================')
            logging.info('Doing: %s' % image_folders_list[train_test][sub_num])

            # ==============================
            # First, get ed_es_diff for this subject
            # ==============================
            for item in os.listdir(image_folders_list[train_test][sub_num][:-8]):
                if 'list.txt' in item:
                    text_file = open(image_folders_list[train_test][sub_num][:-8] + item, "r") 
                    slice_ids_with_annotations = []
                    for l in text_file.readlines():
                        slice_ids_with_annotations.append(int(float(l[-25:-21])))
                    text_file.close()
                    
            for slice_id in np.unique(slice_ids_with_annotations):
                if slice_id % 20 != 0:
                    ed_es_diff = int(slice_id % 20)
                    continue

            # ==============================
            # read image and label
            # ==============================
            image_ED, image_ES, px, py, pz = read_image(image_folders_list[train_test][sub_num], ed_es_diff, nifti_available=False)
            label_ED, label_ES = read_label(label_folders_list[train_test][sub_num], image_ED.shape, ed_es_diff, nifti_available=False)

            img_ED = image_ED.copy()
            img_ES = image_ES.copy()
            lab_ED = label_ED.copy()
            lab_ES = label_ES.copy()

            # ============
            # normalize the image to be between 0 and 1
            # ============
            img_ED = utils.normalise_image(img_ED, norm_type='div_by_max')
            img_ES = utils.normalise_image(img_ES, norm_type='div_by_max')

            pixel_size = (px, py, pz)
            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1]]

            # ============
            # rescale and write to hdf5 the ED image and label
            # ============
            for zz in range(img_ED.shape[2]):

                slice_rescaled_ED = transform.rescale(np.squeeze(img_ED[:, :, zz]),
                                                      scale_vector,
                                                      order = 1,
                                                      preserve_range = True,
                                                      multichannel = False,
                                                      mode = 'constant')
                
                label_rescaled_ED = transform.rescale(np.squeeze(lab_ED[:, :, zz]),
                                                      scale_vector,
                                                      order = 0,
                                                      preserve_range = True,
                                                      multichannel = False,
                                                      mode = 'constant')

                slice_cropped_ED = utils.crop_or_pad_slice_to_size(slice_rescaled_ED, nx, ny)
                label_cropped_ED = utils.crop_or_pad_slice_to_size(label_rescaled_ED, nx, ny)
                
                # ============
                # rotation by 90 degrees
                # ============
                slice_cropped_ED = np.rot90(slice_cropped_ED, k=-1)
                label_cropped_ED = np.rot90(label_cropped_ED, k=-1)

                image_list[train_test].append(slice_cropped_ED)
                label_list[train_test].append(label_cropped_ED)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, image_list, label_list, counter_from, counter_to)
                    _release_tmp_memory(image_list, label_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0
                    
            # ============
            # rescale and write to hdf5 the ES image and label
            # ============
            for zz in range(img_ES.shape[2]):

                slice_rescaled_ES = transform.rescale(np.squeeze(img_ES[:, :, zz]),
                                                      scale_vector,
                                                      order = 1,
                                                      preserve_range = True,
                                                      multichannel = False,
                                                      mode = 'constant')
                
                label_rescaled_ES = transform.rescale(np.squeeze(lab_ES[:, :, zz]),
                                                      scale_vector,
                                                      order = 0,
                                                      preserve_range = True,
                                                      multichannel = False,
                                                      mode = 'constant')

                slice_cropped_ES = utils.crop_or_pad_slice_to_size(slice_rescaled_ES, nx, ny)
                label_cropped_ES = utils.crop_or_pad_slice_to_size(label_rescaled_ES, nx, ny)
                
                # ============
                # rotation by 90 degrees
                # ============
                slice_cropped_ES = np.rot90(slice_cropped_ES, k=-1)
                label_cropped_ES = np.rot90(label_cropped_ES, k=-1)

                image_list[train_test].append(slice_cropped_ES)
                label_list[train_test].append(label_cropped_ES)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, image_list, label_list, counter_from, counter_to)
                    _release_tmp_memory(image_list, label_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0

        # ============
        # write remaining data
        # ============
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

# ====================================================
# Each patient examination typically includes between 200 and 280 images, with 20 images per cardiac cycle.
# Manual delineation is performed on ED and ES images, yielding around 15 manually segmented images per patient.
# ====================================================    
# Images are named in this format :
# P[TWO DIGIT NB]-[FOUR DIGIT NUMBER].dcm
# TWO DIGIT NB: patient number
# FOUR DIGIT NUMBER: it signifies the position of the image within the series.
# For example, P01-0000  would be the first (ED) phase of the slice closest to the base.
# Taking into account that there are twenty phases per slice, you can then determine that an image with the number 0121 would represent the second phase of the sixth slice from the bottom.
# ED images are P01-0000.dcm, P01-0020.dcm, P01-0040.dcm, etc.
# ES images may be for example, depending on the ES definition, P01-0008.dcm, P01-0028.dcm, P01-0048.dcm, etc.
    
# Contours are named in this format:
# P[TWO DIGIT NB]-[FOUR DIGIT NUMBER]-[i/o]contour-[manual/auto].txt
# TWO DIGIT NB: patient number
# FOUR DIGIT NUMBER: position of the image within the series (please see image naming)
# i/o: it signifies the area that the contour represents.
       # I contours are inner contours, or contours that segment the endocardium. (only extrract this, as ACDC labels are for the RV endocardium)
       # O contours are outer contours, contours that segment the epicardium.
# manual/auto: it signifies the process through which the contours were obtained.
       # All of the expert contours will be denoted manual as they were drawn by humans, and all of the contours that were algorithmically generated should be labelled auto (for automatic).
# ====================================================