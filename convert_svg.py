import matplotlib.colors as colors
import matplotlib.cm as cmx
from pathlib import Path
import imageio
import shutil
import subprocess
import tempfile
import matplotlib.pyplot as plt
import numpy as np



def render_svg(bins, node_num):
    #print('bins: ', bins)
    
    cmap = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=len(bins) - 1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    
    first_svg = True
    _, tgt_svg_path = tempfile.mkstemp(suffix='.svg')
    tgt_svg_path = Path(tgt_svg_path)
    
    for i, valid_pred_bin in enumerate(bins):
        _, tmp_bmp_path = tempfile.mkstemp(suffix='.bmp')
        tmp_bmp_path = Path(tmp_bmp_path)
        
        cond = valid_pred_bin # == True) #& (image.numpy().mean(axis=0) < 0.5)
        tmp_bmp_img = np.where(cond, 255, 0)
        imageio.imsave(str(tmp_bmp_path), tmp_bmp_img.astype(np.uint8))
        color = np.asarray(cscalarmap.to_rgba(i))
        color *= 255
        color_hex = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in color])
        exe = "/tmp/potrace-1.16.linux-x86_64/potrace"
        subprocess.call([exe, '-s', '-i', '-C' + color_hex, tmp_bmp_path])
        tmp_bmp_path = Path(tmp_bmp_path)
        tmp_svg_path = tmp_bmp_path.with_suffix(".svg")
        

        if first_svg:
            shutil.move(str(tmp_svg_path), str(tgt_svg_path))
            first_svg = False
        else:
            with tgt_svg_path.open("r") as f_tgt:
                tgt_svg = f_tgt.read()
            with tmp_svg_path.open("r") as f_src:
                src_svg = f_src.read()

            path_start = src_svg.find('<g')
            path_end = src_svg.find('</svg>')

            insert_pos = tgt_svg.find('</svg>')
            tgt_svg = tgt_svg[:insert_pos] + \
                        src_svg[path_start:path_end] + tgt_svg[insert_pos:]
            with tgt_svg_path.open("w") as f_tgt:
                f_tgt.write(tgt_svg)
            tmp_svg_path.unlink()
        tmp_bmp_path.unlink()

    # set opacity 0.5 to see overlaps
    with tgt_svg_path.open("r") as f_tgt:
        tgt_svg = f_tgt.read()
    insert_pos = tgt_svg.find('<g')
    tgt_svg = tgt_svg[:insert_pos] + \
              '<g fill-opacity="0.5">' + tgt_svg[insert_pos:]
    insert_pos = tgt_svg.find('</svg>')
    tgt_svg = tgt_svg[:insert_pos] + '</g>' + tgt_svg[insert_pos:]
    with tgt_svg_path.open("w") as f_tgt:
        f_tgt.write(tgt_svg)
    return tgt_svg_path