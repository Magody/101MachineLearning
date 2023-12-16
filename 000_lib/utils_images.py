from io import BytesIO
from textwrap import wrap
from time import sleep
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm as tqdmnb

def request_url_image(url):
    # Adding information about user agent to don't get blocked
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)    
    # calling urlretrieve function to get resource
    img = Image.open(BytesIO(urllib.request.urlopen(url).read()))

    return img

def plot_imgs_grid(rows,cols,titles,images,figsize=(36,28)):

    f, axarr = plt.subplots(rows, cols, figsize=figsize) # (width,height)

    counter = 0
    for i in range(rows):
        for j in range(cols):
            title = titles[counter].replace("$", "d")  # matplotlib exception with $
            try:
                title = "\n".join(wrap(title, width=30))
            except:
                pass
            axarr[i,j].imshow(images[counter])
            axarr[i,j].set_title(title,fontsize=20)
            counter += 1
    plt.show()
    
def parse_title_img(df:pd.DataFrame):
    """Extract titles and images from dataframes as lists

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    titles = df["title"].values
    imgs = df["cover"].values
    return titles, imgs

def requests_imgs_as_b(imgs:list, time_wait:float=0.1):
    """Request url to Image object

    Args:
        imgs (list): _description_

    Returns:
        _type_: _description_
    """
    imgs_b = []
    for i in tqdmnb(range(len(imgs))):
        imgs_b.append(request_url_image(imgs[i]))
        sleep(time_wait)
    return imgs_b
