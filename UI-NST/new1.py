import ui_nst_image as img

rgb = [0, 0, 0]
rgb[0], rgb[1], rgb[2] = img.HTML2RGB("#00B7FF")
print(rgb)

img.generate_bkg_img(rgb)