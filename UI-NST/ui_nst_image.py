import cv2
import numpy as np


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (400, 300))
    return img


def generate_bkg_img(rgb):
    img = np.ones((20, 40, 3))
    img[:, :, 0] = rgb[0]
    img[:, :, 1] = rgb[1]
    img[:, :, 2] = rgb[2]

    cv2.imshow('img', img)
    cv2.imwrite("con_button2.jpg", img)
    cv2.waitKey(0)


def HTML2RGB(html_color):
    """ convert #RRGGBB to an (R, G, B) tuple """
    html_color = html_color.strip()
    if html_color[0] == '#':
        html_color = html_color[1:]
        r, g, b = html_color[:2], html_color[2:4], html_color[4:]
        r, g, b = [int(n, 16) for n in (r, g, b)]
    return r, g, b


def RGB2HTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor

