import SimpleITK as sitk
import numpy as np
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