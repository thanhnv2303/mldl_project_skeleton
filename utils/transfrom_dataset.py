import os
import shutil

data_folder = "data"
with open(f'{data_folder}/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'{data_folder}/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'{data_folder}/tiny-imagenet-200/val/images/{fn}', f'{data_folder}/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('{data_folder}/tiny-imagenet-200/val/images')