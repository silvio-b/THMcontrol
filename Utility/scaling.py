def ScalingMinMax(mat, mins, maxs, min_val, max_val):
    scaled_mat = (max_val - min_val)*((mat - mins)/(maxs-mins))+min_val
    return scaled_mat

def RevScalingMinMax(scaled_mat, mins, maxs, min_val, max_val):
    mat = (scaled_mat - min_val)*((maxs-mins)/(max_val - min_val)) +mins
    return mat
