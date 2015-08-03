'''
Created on 28 Jul 2015

@author: navjotkukreja
'''

class feature:
    def __init__(self):
        pass
    def extract_features(self, images):
        images = Parallel(n_jobs=12)(delayed(process_image)(i) for i in images)
        return images
    
    
    
counter = Value(c_int)  # defaults to 0
counter_lock = Lock()
def increment(obj, image_path):
    with counter_lock:
        counter.value += 1
    obj.update_progress(image_path, counter)
    
def process_image(image):
    image=(image*255).astype(np.uint8)
    fd = obj.meta(image)
    #increment(obj, image_path)
    return fd