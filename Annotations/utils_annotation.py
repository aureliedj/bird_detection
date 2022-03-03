import cv2 

def getListImg(path):
    
    """Opens an AVI file and convert to list of images.
    
    Inputs:
        path (str): file path

    Returns:
        frames (list of np arrays): list of the 1020 images in the timelapse
        count (int): number of images retrieved in the videofile
    """
    
    cap = cv2.VideoCapture(path)
    
    frames = []
    success = 1
    count = 0
    speed = 1

    while success:
        success, image = cap.read()
        if(count % speed == 0):
            frames.append(image)
        count += 1
    return frames, count