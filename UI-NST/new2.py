def HTMLColorToRGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple """

    colorstring = colorstring.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]
        r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
        r, g, b = [int(n, 16) for n in (r, g, b)]
    return r, g, b


def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor


rgb = HTMLColorToRGB("#F0F8FF")
print(RGBToHTMLColor(rgb))