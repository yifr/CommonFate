import os
import argparse

# NOTE: Background image size 1020 x 1080, object texture size: 1024 x 1024
def main():
    parser = argparse.ArgumentParser(description='Create an image with random black dots')

    parser.add_argument('--channel', type=str, default='RG', help='Input channel(s), eg; R, RG, RGB ')
    parser.add_argument('--threshold', type=str, default='0.1%', help='Threshold for black dot cutoff, eg; 0.5%, 1%')
    parser.add_argument('--blur', type=str, default='0x5', help='how much to blur pixels (larger value causes larger dots, eg; 0x.1, 0x2')
    parser.add_argument('--contrast_stretch', type=str, default='35x95%', help='Clip blurred out pixels (sharpens the image), eg; 5x20%, 1x95%')
    parser.add_argument('--output', type=str, default='random_out.png', help='output filename.png')
    parser.add_argument('--size', type=str, default='1024x1024')
    args = parser.parse_args()

    random_dots(output=args.output, channel=args.channel, threshold=args.threshold, blur=args.blur, contrast_stretch=args.contrast_stretch)

def random_dots(output='texture.jpg', channel='RG', threshold='.1%', blur='0x5', contrast_stretch='35x95%', size='1024x1024'):
    cmd = 'convert -size {} xc: +noise Random -channel {} -threshold {} \
               -channel G -separate +channel \
              \( +clone \) -compose multiply -flatten \
              -blur {} -contrast-stretch {} \
              {}'.format(size, channel, threshold, blur, contrast_stretch, output)

    os.system(cmd)

if __name__=='__main__':
    main()
