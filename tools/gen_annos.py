
import os
import os.path as osp
import random


def gen_coco():
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    root_path = './coco'
    save_path = './datasets/coco/'
    for mode in ('train', 'val'):
        im_root = osp.join(root_path, f'images/{mode}2017')
        lb_root = osp.join(root_path, f'annotations/panoptic_{mode}2017')

        ims = os.listdir(im_root)
        lbs = os.listdir(lb_root)

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{mode}2017/{name}.jpg,annotations/panoptic_{mode}2017/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

def gen_ADE():

    root_path = './'
    save_path = './datasets/ADE/'
    for mode in ('training', 'validation'):
        im_root = osp.join(root_path, f'images/{mode}')
        lb_root = osp.join(root_path, f'annotations/{mode}')

        ims = []
        # print(im_root)
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    ims.append(os.path.join(root, file).replace('\\','/'))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('.png')):
                    lbs.append(os.path.join(root, file).replace('\\','/'))


        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)


        # print(len(ims))
        # print(len(lbs))
        print(ims[0])
        print(lbs[0])
        im_names = [el.split('/')[-1].replace('.jpg', '') for el in ims]
        lb_names = [el.split('/')[-1].replace('.png', '') for el in lbs]
        # common_names = list(set(im_names) & set(lb_names))

        lines = []
        for i, im_lb in enumerate(zip(im_names, lb_names)):
            # print (im_lb)
            im, lb = im_lb
            # im = im[0]
            # lb = lb[0]
            if im == lb:
                lines.append(f'{ims[i]},{lbs[i]}')


        # print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

def gen_ADE20k():

    root_path = './ADE20K_2016_07_26'
    save_path = './datasets/ADE20K/'
    for mode in ('training', 'validation'):
        im_root = osp.join(root_path, f'images/{mode}')
        lb_root = osp.join(root_path, f'images/{mode}')

        ims = []
        # print(im_root)
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.jpg')):
                    ims.append(os.path.join(root, file).replace('\\','/'))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('_seg.png')):
                    lbs.append(os.path.join(root, file).replace('\\','/'))


        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)


        # print(len(ims))
        # print(len(lbs))
        print(ims[0])
        print(lbs[0])
        im_names = [el.split('/')[-1].replace('.jpg', '') for el in ims]
        lb_names = [el.split('/')[-1].replace('_seg.png', '') for el in lbs]
        # common_names = list(set(im_names) & set(lb_names))

        lines = []
        for i, im_lb in enumerate(zip(im_names, lb_names)):
            # print (im_lb)
            im, lb = im_lb
            # im = im[0]
            # lb = lb[0]
            if im == lb:
                lines.append(f'{ims[i]},{lbs[i]}')

        # print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

def gen_sunrgbd():

    root_path = './sunrgb'
    save_path = './datasets/sunrgbd/'
    for mode in ('train', 'test'):
        im_root = osp.join(root_path, f'image/{mode}')
        lb_root = osp.join(root_path, f'label38/{mode}')

        ims = []
        # print(im_root)
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.jpg')):
                    ims.append(os.path.join(root, file).replace('\\','/'))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('.png')):
                    lbs.append(os.path.join(root, file).replace('\\','/'))


        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)


        # print(len(ims))
        # print(len(lbs))
        # print(ims[0])
        # print(lbs[0])
        im_names = [el.replace('img-', '00').split('/')[-1].replace('.jpg', '') for el in ims]
        lb_names = [el.split('/')[-1].replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = []
        # for i, im_lb in enumerate(zip(im_names, lb_names)):
        #     # print (im_lb)
        #     im, lb = im_lb
        #     # im = im[0]
        #     # lb = lb[0]
        #     if im == lb:
        #         lines.append(f'{ims[i]},{lbs[i]}')
        for name in common_names:
            lines.append(f'{im_root}/img-{name[2:]}.jpg,{lb_root}/{name}.png')


        # print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

def gen_voc():

    root_path = './VOC'
    save_path = './datasets/voc/'
    
    im_root = osp.join(root_path, f'JPEGImages/')
    lb_root = osp.join(root_path, f'SegmentationClassAug/')

    ims = []
    # print(im_root)
    for root, dirs, files in os.walk(im_root):
        for file in files:
            if file.lower().endswith(('.jpg')):
                ims.append(os.path.join(root, file).replace('\\','/'))

    lbs = []
    for root, dirs, files in os.walk(lb_root):
        for file in files:
            if file.lower().endswith(('.png')):
                lbs.append(os.path.join(root, file).replace('\\','/'))


    # ims = os.listdir(im_root)
    # lbs = os.listdir(lb_root)


    # print(len(ims))
    # print(len(lbs))
    print(ims[0])
    print(lbs[0])
    im_names = [el.split('/')[-1].replace('.jpg', '') for el in ims]
    lb_names = [el.split('/')[-1].replace('.png', '') for el in lbs]
    # common_names = list(set(im_names) & set(lb_names))
    print (im_names[0])
    print(lb_names[0])
    lines = []
    for i, im_lb in enumerate(zip(im_names, lb_names)):
        # print (im_lb)
        im, lb = im_lb
        # im = im[0]
        # lb = lb[0]
        if im == lb:
            lines.append(f'{ims[i]},{lbs[i]}')

    # print(lines[0])
    # lines = [
    #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
    #     for name in common_names
    # ]

    random.shuffle(lines)
        # print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

    total_len = len(lines)
    num_train = int(total_len * 0.8)
    # num_val = total_len - num_train
    with open(f'{save_path}train.txt', 'w') as fw:
        fw.write('\n'.join(lines[:num_train]))

    with open(f'{save_path}val.txt', 'w') as fw:
        fw.write('\n'.join(lines[:num_train]))

def gen_mapillary():

    root_path = './'
    save_path = './datasets/mapi/'
    for mode in ('training', 'validation'):
        im_root = osp.join(root_path, f'{mode}/images')
        lb_root = osp.join(root_path, f'{mode}/v2.0/labels')

        ims = []
        # print(im_root)
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.jpg')):
                    ims.append(os.path.join(root, file).replace('\\','/'))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('.png')):
                    lbs.append(os.path.join(root, file).replace('\\','/'))

        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)

        # print(len(ims))
        # print(len(lbs))
        print(ims[0])
        print(lbs[0])
        im_names = [el.split('/')[-1].replace('.jpg', '') for el in ims]
        lb_names = [el.split('/')[-1].replace('.png', '') for el in lbs]
        # common_names = list(set(im_names) & set(lb_names))

        lines = []
        for i, im_lb in enumerate(zip(im_names, lb_names)):
            # print (im_lb)
            im, lb = im_lb
            # im = im[0]
            # lb = lb[0]
            if im == lb:
                lines.append(f'{ims[i]},{lbs[i]}')

        # print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

def gen_IDD():
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    root_path = './IDD_Segmentation'
    save_path = './datasets/IDD/'
    for mode in ('train', 'val'):
        im_root = osp.join(root_path, f'leftImg8bit/{mode}')
        lb_root = osp.join(root_path, f'gtFine/{mode}')

        ims = []
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    ims.append(os.path.join(root, file))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('.json')):
                    lbs.append(os.path.join(root, file))


        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)


        print(len(ims))
        print(len(lbs))
        print(ims[0])
        print(lbs[0])
        im_names = [el.split('_')[1].split('\\')[-2:] for el in ims]
        lb_names = [el.split('_')[1].split('\\')[-2:] for el in lbs]
        # common_names = list(set(im_names) & set(lb_names))

        lines = []
        for i, im_lb in enumerate(zip(im_names, lb_names)):
            # print (im_lb)
            im, lb = im_lb
            im = im[0]+im[1]
            lb = lb[0]+lb[1]
            if im == lb:
                lines.append(f'{ims[i]},{lbs[i]}')

        print(lines[0])
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))


def gen_BDD():
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    root_path = './'
    save_path = './datasets/bdd100k/'
    for mode in ('train', 'val'):
        im_root = osp.join(root_path, f'seg/images/{mode}')
        lb_root = osp.join(root_path, f'seg/labels/{mode}')

        ims = []
        for root, dirs, files in os.walk(im_root):
            for file in files:
                if file.lower().endswith(('.jpg')):
                    ims.append(os.path.join(root, file).replace('\\','/'))

        lbs = []
        for root, dirs, files in os.walk(lb_root):
            for file in files:
                if file.lower().endswith(('.png')):
                    lbs.append(os.path.join(root, file).replace('\\','/'))


        # ims = os.listdir(im_root)
        # lbs = os.listdir(lb_root)

        # print(len(ims))
        # print(len(lbs))
        
        im_names = [el.split('/')[-1].replace(".jpg", "") for el in ims]
        lb_names = [el.split('/')[-1].replace("_train_id.png", "") for el in lbs]
        
        # common_names = list(set(im_names) & set(lb_names))

        lines = []
        for i, im_lb in enumerate(zip(im_names, lb_names)):
            # print (im_lb)
            im, lb = im_lb
            # im = im[0]
            # lb = lb[0]
            if im == lb:
                lines.append(f'{ims[i]},{lbs[i]}')
                
        # lines = [
        #     f'leftImg8bit/{name}.jpg,labels/{mode}2017/{name}.png'
        #     for name in common_names
        # ]

        with open(f'{save_path}{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))

gen_sunrgbd()
# gen_voc()
