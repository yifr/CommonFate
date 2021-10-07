"""
Extract frame differences and write out features to file
"""

def frame_diff(images):
    """Returns differences of successive frame given input array of images
    input:
        images: t x h x w ndarray of images
    Output:
        frame_differences: (t - 1) x h x w
    """
    if len(images.shape) < 3:
        raise ValueError(f"images shape was {images.shape}, needs to be n_frames x width x height")
    n_frames, height, width = images.shape
    differences = np.zeros(n_frames - 1, height, width)

    i = 1
    while i < n_frames:
        differences[i - 1] = images[i] - images[i - 1]
