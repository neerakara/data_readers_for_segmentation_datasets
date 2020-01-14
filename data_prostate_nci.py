import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import utils
import pydicom as dicom
import nrrd
import subprocess
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         cv_fold_number):
    
    if cv_fold_number == 1:
        if patient_id < 16: return 'train'
        elif patient_id < 21: return 'validation'
        else: return 'test'

# ===============================================================
# ===============================================================
def count_slices(input_folder,
                 folder_base,
                 cv_fold_number):

    num_slices = {'train': 0, 'test': 0, 'validation': 0}

    for folder in os.listdir(input_folder):
        
        if not folder.startswith(folder_base):
            continue

        patient_id = int(folder.split('-')[-1])
        
        if patient_id > 30:
            continue

        path = os.path.join(input_folder, folder)
        
        for dirName, subdirList, fileList in os.walk(path):
        
            for filename in fileList:
            
                if filename.lower().endswith('.dcm'):  # check whether the file's DICOM

                    train_test = test_train_val_split(patient_id, cv_fold_number)

                    num_slices[train_test] += 1

    return num_slices

# ===============================================================
# ===============================================================
def get_patient_folders(input_folder,
                        folder_base,
                        cv_fold_number):

    folder_list = {'train': [], 'test': [], 'validation': []}

    for folder in os.listdir(input_folder):
    
        if folder.startswith('Prostate3T-01'):
        
            patient_id = int(folder.split('-')[-1])
            
            if patient_id > 30:
                continue
            
            train_test = test_train_val_split(patient_id, cv_fold_number)
            
            folder_list[train_test].append(os.path.join(input_folder, folder))

    return folder_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 output_file,
                 size,
                 target_resolution,
                 cv_fold_num):

    # =======================
    # =======================
    image_folder = os.path.join(input_folder, 'Prostate-3T')
    mask_folder = os.path.join(input_folder, 'NCI_ISBI_Challenge-Prostate3T_Training_Segmentations')

    # =======================
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # =======================
    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(image_folder,
                                      folder_base='Prostate3T-01',
                                      cv_fold_number = cv_fold_num)
    
    num_slices = count_slices(image_folder,
                              folder_base='Prostate3T-01',
                              cv_fold_number = cv_fold_num)
    
    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']

    # =======================
    # =======================
    print('Debug: Check if sets add up to correct value:')
    print(n_train, n_val, n_test, n_train + n_val + n_test)

    # =======================
    # Create datasets for images and masks
    # =======================
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}
    nx_list = {'test': [], 'train': [], 'validation': []}
    ny_list = {'test': [], 'train': [], 'validation': []}
    nz_list = {'test': [], 'train': [], 'validation': []}
    px_list = {'test': [], 'train': [], 'validation': []}
    py_list = {'test': [], 'train': [], 'validation': []}
    pz_list = {'test': [], 'train': [], 'validation': []}
    pat_names_list = {'test': [], 'train': [], 'validation': []}

    # =======================
    # =======================
    logging.info('Parsing image files')
    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0

        patient_counter = 0

        for folder in folder_list[train_test]:

            patient_counter += 1

            logging.info('================================')
            logging.info('Doing: %s' % folder)
            pat_names_list[train_test].append(str(folder.split('-')[-1]))

            lstFilesDCM = []  # create an empty list
            
            for dirName, subdirList, fileList in os.walk(folder):
            
                # fileList.sort()
                for filename in fileList:
                
                    if ".dcm" in filename.lower():  # check whether the file's DICOM
                        lstFilesDCM.append(os.path.join(dirName, filename))

            # Get ref file
            RefDs = dicom.read_file(lstFilesDCM[0])

            # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

            # Load spacing values (in mm)
            pixel_size = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
            px_list[train_test].append(float(RefDs.PixelSpacing[0]))
            py_list[train_test].append(float(RefDs.PixelSpacing[1]))
            pz_list[train_test].append(float(RefDs.SliceThickness))

            print('PixelDims')
            print(ConstPixelDims)
            print('PixelSpacing')
            print(pixel_size)

            # The array is sized based on 'ConstPixelDims'
            img = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

            # loop through all the DICOM files
            for filenameDCM in lstFilesDCM:

                # read the file
                ds = dicom.read_file(filenameDCM)

                # ======
                # store the raw image data
                # img[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
                # index number field is not set correctly!
                # instead instance number is the slice number.
                # ======
                img[:, :, ds.InstanceNumber - 1] = ds.pixel_array
                
            # ================================
            # save as nifti, this sets the affine transformation as an identity matrix
            # ================================    
            nifti_img_path = lstFilesDCM[0][:lstFilesDCM[0].rfind('/')+1]
            utils.save_nii(img_path = nifti_img_path + 'img.nii.gz', data = img, affine = np.eye(4))
    
            # ================================
            # do bias field correction
            # ================================
            input_img = nifti_img_path + 'img.nii.gz'
            output_img = nifti_img_path + 'img_n4.nii.gz'
            subprocess.call(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", input_img, output_img])
    
            # ================================    
            # read bias corrected image
            # ================================    
            img = utils.load_nii(img_path = nifti_img_path + 'img_n4.nii.gz')[0]

            # ================================    
            # normalize the image
            # ================================    
            img = utils.normalise_image(img, norm_type='div_by_max')

            # ================================    
            # read the labels
            # ================================    
            mask_path = os.path.join(mask_folder, folder.split('/')[-1] + '.nrrd')
            mask, options = nrrd.read(mask_path)

            # fix swap axis
            mask = np.swapaxes(mask, 0, 1)
            
            # ================================
            # save as nifti, this sets the affine transformation as an identity matrix
            # ================================    
            utils.save_nii(img_path = nifti_img_path + 'lbl.nii.gz', data = mask, affine = np.eye(4))
            
            nx_list[train_test].append(mask.shape[0])
            ny_list[train_test].append(mask.shape[1])
            nz_list[train_test].append(mask.shape[2])

            print('mask.shape')
            print(mask.shape)
            print('img.shape')
            print(img.shape)

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):

                slice_img = np.squeeze(img[:, :, zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   multichannel=False,
                                                   mode = 'constant')

                slice_mask = np.squeeze(mask[:, :, zz])
                mask_rescaled = transform.rescale(slice_mask,
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode='constant')

                slice_cropped = utils.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                mask_cropped = utils.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                img_list[train_test].append(slice_cropped)
                mask_list[train_test].append(mask_cropped)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0


        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)

    # Write the small datasets
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patnames_%s' % tt, data=np.asarray(pat_names_list[tt], dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         train_test,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list, mask_list, train_test):
    
    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()

# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                force_overwrite=False,
                                cv_fold_num = 1):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_2d_size_%s_res_%s_cv_fold_%d.hdf5' % (size_str, res_str, cv_fold_num)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     size,
                     target_resolution,
                     cv_fold_num)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    cv_fold_num,
                                    train_test,
                                    idx):
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================
    image_folder = os.path.join(input_folder, 'Prostate-3T')
    label_folder = os.path.join(input_folder, 'NCI_ISBI_Challenge-Prostate3T_Training_Segmentations')
    folder_list = get_patient_folders(image_folder,
                                      folder_base='Prostate3T-01',
                                      cv_fold_number = cv_fold_num)
    folder = folder_list[train_test][idx]

    # ==================
    # make a list of all dcm images for this subject
    # ==================                        
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(folder):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))
                
    # ==================
    # read bias corrected image
    # ==================
    nifti_img_path = lstFilesDCM[0][:lstFilesDCM[0].rfind('/')+1]
    image = utils.load_nii(img_path = nifti_img_path + 'img_n4.nii.gz')[0]

    # ============
    # normalize the image to be between 0 and 1
    # ============
    image = utils.normalise_image(image, norm_type='div_by_max')

    # ==================
    # read the label file
    # ==================        
    label = utils.load_nii(img_path = nifti_img_path + 'lbl.nii.gz')[0]
    
    return image, label