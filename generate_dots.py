import os 
import argparse

parser = argparse.ArgumentParser(description='Create an image with random black dots')

parser.add_argument('--channel', type=str, default='RG', help='Input channel(s), eg; R, RG, RGB ')
parser.add_argument('--threshold', type=str, default='0.1%', help='Threshold for black dot cutoff, eg; 0.5%, 1%')
parser.add_argument('--blur', type=str, default='0x2', help='how much to blur pixels (larger value causes larger dots, eg; 0x.1, 0x2')
parser.add_argument('--contrast_stretch', type=str, default='50%', help='Clip blurred out pixels (sharpens the image), eg; 5x20%, 1x95%')
parser.add_argument('--output', type=str, default='random_out.png', help='output filename.png')
args = parser.parse_args()
print(args)
cmd = 'convert -size 1024x1024 xc: +noise Random -channel {} -threshold {} \
           -channel G -separate +channel \
          \( +clone \) -compose multiply -flatten \
          -blur {} -contrast-stretch {} \
          {}'.format(args.channel, args.threshold, args.blur, args.contrast_stretch, args.output)
print(cmd)

os.system(cmd)

