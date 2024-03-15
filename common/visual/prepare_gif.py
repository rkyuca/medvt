import glob
import os
import sys

from PIL import Image


def make_gif(frame_folder, outfile):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.jpeg")]
    if len(frames) == 0:
        frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.jpg")]

    frame_one = frames[0]
    frame_one.save(outfile, format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


'''
if __name__ == "__main__":
    input_path = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/supplement_video/moca/arabian_horn_viper_rgb_res1/'
    output_path = '/local/riemann/home/rezaul/projects/transformer-vos-main/results/supplement_video/moca/arabian_horn_viper_feats_gif/'
    
    sub_dirs = os.listdir(input_path)
    for dn in sub_dirs:
        if not os.path.isdir(os.path.join(input_path, dn)):
            continue
        print('current dir:%s' % dn)
        out_file = os.path.join(output_path, '%s.gif' % dn)
        make_gif(os.path.join(input_path, dn), out_file)
'''

if __name__ == "__main__":
    make_gif(sys.argv[1], sys.argv[2])
