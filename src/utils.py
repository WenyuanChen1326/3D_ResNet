import SimpleITK as sitk
import numpy as np
import cc3d
import matplotlib.pyplot as plt
# def scale_up_block(input_block, new_resol = [224,224,224], interpolation = 'cubic'):
#     # Convert the input numpy array to a SimpleITK Image
#     input_image = sitk.GetImageFromArray(input_block.astype(np.float32))
#     # Define the new size
#     new_size = new_resol
#     # Compute the scaling factors for each dimension
#     scaling_factors = [float(new_size[0]) / input_image.GetSize()[0], 
#                        float(new_size[1]) / input_image.GetSize()[1], 
#                        float(new_size[2]) / input_image.GetSize()[2]]
#     # Create a resampling filter
#     resample_filter = sitk.ResampleImageFilter()
#     # Set the output image size
#     resample_filter.SetSize(new_size)

#     # Set the interpolator. 
#     if interpolation == 'cubic':
#         resample_filter.SetInterpolator(sitk.sitkBSpline)
#     # For cubic interpolation, change to sitk.sitkBSpline.
#     if interpolation == 'linear':
#         resample_filter.SetInterpolator(sitk.sitkLinear)
#     # Calculate new spacing based on the scaling factors
#     original_spacing = input_image.GetSpacing()
#     new_spacing = [original_spacing[0] / scaling_factors[0], 
#                    original_spacing[1] / scaling_factors[1], 
#                    original_spacing[2] / scaling_factors[2]]
#     resample_filter.SetOutputSpacing(new_spacing)
#     # Set the output origin, to keep it same as input
#     resample_filter.SetOutputOrigin(input_image.GetOrigin())

#     # Set the output direction, to keep it same as input
#     resample_filter.SetOutputDirection(input_image.GetDirection())

#     # Perform the resampling
#     resampled_image = resample_filter.Execute(input_image)

#     # Convert the resampled SimpleITK image to a NumPy array
#     resampled_array = sitk.GetArrayFromImage(resampled_image)
#     # print(resampled_array.shape)

#     # Transpose the array to match the conventional (height, width, channels) format
#     resampled_array_np = np.transpose(resampled_array, (1, 2, 0))
#     # print(resampled_array_np.shape)

#     # Return the resampled array
#     return resampled_array_np

def sample_neg_block(seg_data, local_max_points_coordinate_without_pos_coordinate, block_size=(3, 3, 3), sample_size=10, negative=True):
    if len(local_max_points_coordinate_without_pos_coordinate) == 0:
        return []
    max_attempts = 100
    # attempt = 0
    # block_found = False
    adjusted_smaple_size = min(sample_size, len(local_max_points_coordinate_without_pos_coordinate))
    neg_output_coord_list = []
    success_count = 0
    while success_count < adjusted_smaple_size:
        # print(f"success_count: {success_count}")
        attempt = 0
        block_found = False
        while attempt < max_attempts and not block_found:
            # # Generate random starting indices
            # start_x = np.random.randint(0, seg_data.shape[0] - 2)  # -2 to include the end index for a 3x3x3 block
            # start_y = np.random.randint(0, seg_data.shape[1] - 2)
            # start_z = np.random.randint(0, seg_data.shape[2] - 2)
            # index = np.random.choice(len(local_max_points_coordinate_without_pos_coordinate))
            index = np.random.choice(range(len(local_max_points_coordinate_without_pos_coordinate)))
            x_center, y_center, z_center = local_max_points_coordinate_without_pos_coordinate[index]
            # Calculate half sizes for each dimension, adjusting for even sizes
            half_size_x = block_size[0] // 2
            half_size_y = block_size[1] // 2
            half_size_z = block_size[2] // 2
            x_start = max(x_center - half_size_x, 0)
            x_end = min(x_start + block_size[0], seg_data.shape[0])
            y_start = max(y_center - half_size_y, 0)
            y_end = min(y_start + block_size[1], seg_data.shape[1])
            z_start = max(z_center - half_size_z, 0)
            z_end = min(z_start + block_size[2], seg_data.shape[2])

            # Correct the start positions if the block exceeds the mask dimensions
            if x_end - x_start < block_size[0]: x_start = x_end - block_size[0]
            if y_end - y_start < block_size[1]: y_start = y_end - block_size[1]
            if z_end - z_start < block_size[2]: z_start = z_end - block_size[2]
              # Ensure adjustments did not result in negative start values
            x_start, y_start, z_start = max(x_start, 0), max(y_start, 0), max(z_start, 0)
            # Extract the 3x3x3 block
            
            # Check if the block contains only 0s
            block = seg_data[x_start:x_start+block_size[0], y_start:y_start+block_size[1], z_start:z_start+block_size[2]]
            if negative:
                if np.all(block == 0):
                    block_found = True
                    neg_output_coord_list.append((x_start, y_start, z_start))
                    success_count += 1
                    attempt += 1
            else:
                block_found = True
                neg_output_coord_list.append((x_start, y_start, z_start))
                success_count += 1
                attempt += 1

            
    return neg_output_coord_list

def filter_separate_segmentation_mask_by_diameter_and_SUV_max_and_voxel_of_interest(suv_data, voxel_dimensions, separate_segmentation_masks, diameter_in_cm = 6, SUV_max = 3, voxel_of_interst = 3):
    '''
    This function filters the separate segmentation masks by diameter and SUV_max to remove noise ground truth and then obtain tumor with voxel_of_interest
    '''
    # filtered_separate_segmentation_masks = {}
    filtered_separate_segmentation_masks = []
    if len(separate_segmentation_masks) == 0:
        return filtered_separate_segmentation_masks
    for idx, mask in enumerate(separate_segmentation_masks):
        #count the number of voxels in the mask
        num_voxels = np.sum(mask)
        #the volume of each voxel is the product of its dimensions
        tumor_volume = num_voxels * np.prod(voxel_dimensions)/1000
        mask_diameter = np.round(get_diameter_from_sphere_volume(tumor_volume),4)
        mask_SUV_max = suv_data[mask].max()
        # print(f"num_voxels: {num_voxels} tumor volume: {tumor_volume} diameter:{mask_diameter}")
        if (mask_diameter >= diameter_in_cm or mask_SUV_max >= SUV_max) and num_voxels <= voxel_of_interst:
            # filtered_separate_segmentation_masks[idx] = mask
            filtered_separate_segmentation_masks.append(mask)

    return filtered_separate_segmentation_masks
def get_diameter_from_sphere_volume(volume):
    return 2 * (3 * volume / (4 * np.pi)) ** (1/3) 

def get_connected_components_3D(seg_data, connectivity =26):
    # print(seg_data.shape)
    # 4 is a rough estimate of the minimum volume of a tumor in cubic mm
    # minV= calculate_sphere_vol_from_diameter(4) #(removing noise in the segmentation data but not sure what the best number of pixel to deem too small is)
    labels_out = cc3d.connected_components(seg_data, connectivity = connectivity)

    # print(f"labels_out shape: {labels_out.shape}")
    cc_n = np.max(np.unique(labels_out))
    separate_seg_masks = []
    for i in range(1,cc_n+1):
        # print(f"tumor index: {i}")
        # size_n=np.sum(labels_out==i)
        # if size_n<minV:
        #     seg_data[labels_out==i]=0
        # else:
        c_mask = labels_out == i
        separate_seg_masks.append(c_mask)
    return separate_seg_masks

def scale_up_block(input_block, new_resol = [224,224,224],interpolation = 'cubic', spacing = (2.0364201068878174, 2.0364201068878174, 3.0)):
    # Convert the input numpy array to a SimpleITK Image
    input_image = sitk.GetImageFromArray(input_block.astype(np.float32))
    # Define the new size
    new_size = new_resol
    # Compute the scaling factors for each dimension
    scaling_factors = [float(new_size[0]) / input_image.GetSize()[0], 
                       float(new_size[1]) / input_image.GetSize()[1], 
                       float(new_size[2]) / input_image.GetSize()[2]]
    # Create a resampling filter
    resample_filter = sitk.ResampleImageFilter()
    # Set the output image size
    resample_filter.SetSize(new_size)

    # Set the interpolator. 
    if interpolation == 'cubic':
        resample_filter.SetInterpolator(sitk.sitkBSpline)
    # For cubic interpolation, change to sitk.sitkBSpline.
    if interpolation == 'linear':
        resample_filter.SetInterpolator(sitk.sitkLinear)
    # Calculate new spacing based on the scaling factors
    # original_spacing = input_image.GetSpacing()
    original_spacing = spacing
    new_spacing = [original_spacing[0] / scaling_factors[0], 
                   original_spacing[1] / scaling_factors[1], 
                   original_spacing[2] / scaling_factors[2]]
    resample_filter.SetOutputSpacing(new_spacing)
    # Set the output origin, to keep it same as input
    resample_filter.SetOutputOrigin(input_image.GetOrigin())

    # Set the output direction, to keep it same as input
    resample_filter.SetOutputDirection(input_image.GetDirection())

    # Perform the resampling
    resampled_image = resample_filter.Execute(input_image)

    # Convert the resampled SimpleITK image to a NumPy array
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    # print(resampled_array.shape)

    # Transpose the array to match the conventional (height, width, channels) format
    # resampled_array_np = np.transpose(resampled_array, (1, 2, 0))
    # print(resampled_array_np.shape)

    # Return the resampled array
    return resampled_array