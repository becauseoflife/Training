from PIL import Image


def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    f1 = 1.0 * w_box/w
    f2 = 1.0 * h_box/h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    # print(w, 'x', h)
    # print(width, "x", height)
    return pil_image.resize((width, height), Image.ANTIALIAS)