# ==================================================================
# import 
# ==================================================================
import datapaths
import data_brain_hcp
import data_brain_abide
import data_cardiac_acdc
import data_cardiac_rvsc
import data_prostate_nci
import data_prostate_pirad_erc

# ============================
# main function that redirects the data loading to the requested anatomy and dataset
# ============================   
def load_dataset(anatomy,
                 dataset,
                 train_test_validation,
                 first_run = False): # <-- SET TO TRUE FOR THE FIRST RUN OF A PARTICULAR DATASET

    # =====================================
    # brain datasets
    # =====================================
    if anatomy is 'brain':

        # =====================================
        # HCP
        # =====================================
        if (dataset is 'HCP_T1') or (dataset is 'HCP_T2'):
        
            # =====================================
            # subject ids for the train, test and validation subsets.
            # change indices of train-test-validation split, if required.
            # =====================================
            if train_test_validation is 'train':
                idx_start = 0
                idx_end = 20
            
            elif train_test_validation is 'validation':
                idx_start = 20
                idx_end = 25
            
            elif train_test_validation is 'test':
                idx_start = 50
                idx_end = 70
            
            # =====================================
            # for the HCP dataset, both T1 and T2 images are available for each subject
            # =====================================
            if dataset is 'HCP_T1':
                protocol = 'T1'
                
            elif dataset is 'HCP_T2':
                protocol = 'T2'
            
            # =====================================
            # Set the desired resolution and image size here.
            # Currently, the pre-processing is set up for rescaling coronal slices to this target resolution and target image size,
            # while keeping the resolution perpendicular to the coronal slices at its original value.
            # =====================================
            image_size = (256, 256)
            target_resolution = (0.7, 0.7)
            
            # =====================================
            # The image_depth parameter indicates the number of coronal slices in each image volume.
            # All image volumes are cropped from the edges or padded with zeros, in order to ensure that they have the same number of coronal slices,
            # keeping the resolution in this direction the same as in the original images.
            # =====================================
            image_depth = 256
              
            # =====================================
            # pre-processing function that returns a hdf5 handle containing the images and labels
            # =====================================
            data_brain = data_brain_hcp.load_and_maybe_process_data(input_folder = datapaths.orig_dir_hcp,
                                                                    preprocessing_folder = datapaths.preproc_dir_hcp,
                                                                    idx_start = idx_start,
                                                                    idx_end = idx_end,             
                                                                    protocol = protocol,
                                                                    size = image_size,
                                                                    depth = image_depth,
                                                                    target_resolution = target_resolution)
            
        # =====================================
        # ABIDE
        # =====================================
        elif (dataset is 'ABIDE_caltech') or (dataset is 'ABIDE_stanford'):
        
            # =====================================
            # subject ids for the train, test and validation subsets
            # change indices of train-test-validation split, if required.
            # =====================================
            if train_test_validation is 'train':
                idx_start = 0
                idx_end = 10
            
            elif train_test_validation is 'validation':
                idx_start = 10
                idx_end = 15
            
            elif train_test_validation is 'test':
                idx_start = 16
                idx_end = 36
                        
            # =====================================
            # Set the desired resolution and image size here.
            # Currently, the pre-processing is set up for rescaling coronal slices to this target resolution and target image size,
            # while keeping the resolution perpendicular to the coronal slices at its original value.
            # =====================================
            image_size = (256, 256)
            target_resolution = (0.7, 0.7)
            
            # =====================================
            # within the ABIDE dataset, there are multiple T1 datasets - we have preprocessed the data from 2 of these so far.
            # =====================================
            # =====================================
            # The image_depth parameter indicates the number of coronal slices in each image volume.
            # All image volumes are cropped from the edges or padded with zeros, in order to ensure that they have the same number of coronal slices,
            # keeping the resolution in this direction the same as in the original images.
            # =====================================
            if dataset is 'ABIDE_caltech':
                site_name = 'caltech'
                image_depth = 256
                
            elif dataset is 'ABIDE_stanford':
                site_name = 'stanford'
                image_depth = 132
              
            # =====================================
            # pre-processing function that returns a hdf5 handle containing the images and labels
            # =====================================            
            data_brain = data_brain_abide.load_and_maybe_process_data(input_folder = datapaths.orig_dir_abide,
                                                                      preprocessing_folder = datapaths.preproc_dir_abide,
                                                                      site_name = site_name,
                                                                      idx_start = idx_start,
                                                                      idx_end = idx_end,             
                                                                      protocol = 'T1',
                                                                      size = image_size,
                                                                      depth = image_depth,
                                                                      target_resolution = target_resolution,
                                                                      force_overwrite = False)
        
        # =====================================  
        # Extract images and labels from the hdf5 handle and return these.
        # If desired, the original resolutions of the images and original image sizes, as well as patient ids are also stored in the hdf5 file.
        # =====================================  
        images = data_brain['images']
        labels = data_brain['labels']
                        
    # =====================================
    # 
    # =====================================
    elif anatomy is 'cardiac':
        
        data_mode = '2D'
        image_size = (256, 256)
        target_resolution = (1.33, 1.33)
        cv_fold_number = 1

        # =====================================
        # 
        # =====================================
        if dataset is 'ACDC':
                        
            data_cardiac = data_cardiac_acdc.load_and_maybe_process_data(input_folder = datapaths.orig_dir_acdc,
                                                                         preprocessing_folder = datapaths.preproc_dir_acdc,
                                                                         mode = data_mode,
                                                                         size = image_size,
                                                                         target_resolution = target_resolution,
                                                                         cv_fold_num = cv_fold_number)
            
        elif dataset is 'RVSC':

            data_cardiac = data_cardiac_rvsc.load_and_maybe_process_data(input_folder = datapaths.orig_dir_rvsc,
                                                                         preprocessing_folder = datapaths.preproc_dir_rvsc,
                                                                         mode = data_mode,
                                                                         size = image_size,
                                                                         target_resolution = target_resolution,
                                                                         cv_fold_num = cv_fold_number)
            
        
        # =====================================  
        # 
        # =====================================  
        images = data_cardiac['images_' + train_test_validation]
        labels = data_cardiac['labels_' + train_test_validation]
        
    # =====================================
    # 
    # =====================================
    elif anatomy is 'prostate':
        
        image_size = (256, 256)
        target_resolution = (0.625, 0.625)

        # =====================================
        # 
        # =====================================
        if dataset is 'NCI':
            
            cv_fold_number = 1
                    
            data_pros = data_prostate_nci.load_and_maybe_process_data(input_folder = datapaths.orig_dir_nci,
                                                                      preprocessing_folder = datapaths.preproc_dir_nci,
                                                                      size = image_size,
                                                                      target_resolution = target_resolution,
                                                                      cv_fold_num = cv_fold_number)
            
            # =====================================  
            # 
            # =====================================  
            images = data_pros['images_' + train_test_validation]
            labels = data_pros['labels_' + train_test_validation]

        # =====================================
        # 
        # =====================================            
        elif dataset is 'PIRAD_ERC':
            
            pirad_erc_labeller = 'ek'
            
            # =====================================
            # subject ids for the train, test and validation subsets
            # =====================================
            if train_test_validation is 'train':
                idx_start = 40
                idx_end = 68
            
            elif train_test_validation is 'validation':
                idx_start = 20
                idx_end = 40
            
            elif train_test_validation is 'test':
                idx_start = 0
                idx_end = 20
            
            data_pros = data_prostate_pirad_erc.load_data(input_folder = datapaths.orig_dir_pirad_erc,
                                                          preproc_folder = datapaths.preproc_dir_pirad_erc,
                                                          idx_start = idx_start,
                                                          idx_end = idx_end,
                                                          size = image_size,
                                                          target_resolution = target_resolution,
                                                          labeller = pirad_erc_labeller)  

            # =====================================  
            # 
            # =====================================  
            images = data_pros['images']
            labels = data_pros['labels']
            
    return images, labels
