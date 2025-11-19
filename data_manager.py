from __future__ import print_function, absolute_import
import os
import numpy as np
import random
from PIL import Image

def process_eval_rgbnt(data_dir, eval_mode='v2t', trial = 0, relabel=False,imgh=128,imgw=128):
    random.seed(trial)


    test_rgb_path = data_dir + '/R/'
    test_ni_path = data_dir + '/N/'

    test_rgb_list_all = os.listdir(test_rgb_path)  # all id in the rgb part of the whole dataset
    test_nir_list_all = os.listdir(test_ni_path)  # all id in the nir part of the whole dataset

    #############################select odd to construct testing set#################
    # print(test_rgb_list, test_nir_list)
    test_rgb_list = []  # select odd
    for i in range(len(test_rgb_list_all)):
        cur_id = test_rgb_list_all[i]
        cur_id = int(cur_id)  # str2num
        if cur_id % 2 == 0:
            # print(cur_id)
            test_rgb_list.append(test_rgb_list_all[i])
    # print(len(test_rgb_list))
    test_nir_list = []  # select odd
    for i in range(len(test_nir_list_all)):
        cur_id = test_nir_list_all[i]
        cur_id = int(cur_id)  # str2num
        if cur_id % 2 == 0:
            # print(cur_id)
            test_nir_list.append(test_nir_list_all[i])

    test_rgb_name = []
    test_rgb_image = []
    for i in range(len(test_rgb_list)):
        cur_id_path = test_rgb_path + test_rgb_list[i]
        cur_id_list = os.listdir(cur_id_path)
        # print(cur_id_list)
        for j in range(len(cur_id_list)):
            imgname = cur_id_path + '/' + cur_id_list[j]
            if os.path.splitext(imgname)[1] == '.jpg':
                img = Image.open(imgname)
                # print(img.size)
                img = img.resize((imgw, imgh), Image.ANTIALIAS)
                pix_array = np.array(img)
                test_rgb_image.append(pix_array)
                test_rgb_name.append(cur_id_list[j])

    test_rgb_image = np.array(test_rgb_image)
    # print(test_rgb_image.shape)

    test_nir_name = []
    test_nir_image = []
    for i in range(len(test_nir_list)):
        cur_id_path = test_ni_path + test_nir_list[i]
        cur_id_list = os.listdir(cur_id_path)
        for j in range(len(cur_id_list)):
            imgname = cur_id_path + '/' + cur_id_list[j]
            # print(imgname)
            if os.path.splitext(imgname)[1] == '.jpg':
                img = Image.open(imgname)
                img = img.resize((imgw, imgh), Image.ANTIALIAS)
                pix_array = np.array(img)
                test_nir_image.append(pix_array)
                test_nir_name.append(cur_id_list[j])

    test_nir_image = np.array(test_nir_image)

    ################################process person id: tranform 0599 to 149.0##########################
    test_rgb_id = [s.split('_')[0] for s in test_rgb_name]
    test_nir_id = [s.split('_')[0] for s in test_nir_name]
    test_twomodal_id = test_rgb_id + test_nir_id
    unique_id = np.unique(test_twomodal_id)
    test_rgb_label = np.ones(len(test_rgb_id)) * -1
    test_nir_label = np.ones(len(test_nir_id)) * -1
    for i in range(len(unique_id)):
        tmp_rgb = [k for k, v in enumerate(test_rgb_id) if v == unique_id[i]]
        test_rgb_label[tmp_rgb] = i

        tmp_nir = [k for k, v in enumerate(test_nir_id) if v == unique_id[i]]
        test_nir_label[tmp_nir] = i

    ################################process camera id: tranform c0001 to 0.0 ##########################
    test_rgb_cid = [s.split('_')[1][1:] for s in test_rgb_name]
    test_nir_cid = [s.split('_')[1][1:] for s in test_nir_name]
    test_twomodal_cid = test_rgb_cid + test_nir_cid
    unique_cid = np.unique(test_twomodal_cid)
    test_rgb_clabel = np.ones(len(test_rgb_cid)) * -2
    test_nir_clabel = np.ones(len(test_nir_cid)) * -2
    for i in range(len(unique_cid)):
        tmp_rgb = [k for k, v in enumerate(test_rgb_cid) if v == unique_cid[i]]
        test_rgb_clabel[tmp_rgb] = i
        tmp_nir = [k for k, v in enumerate(test_nir_cid) if v == unique_cid[i]]
        test_nir_clabel[tmp_nir] = i



    if eval_mode=='v2t':
        #v--probe/query t---gallery
        print('.....................v2t...................')
        gallery_img = test_nir_image
        gallery_label = test_nir_label
        gallery_clabel = test_nir_clabel
        gallery_name = test_nir_name
        # print(gallery_name[0],  gallery_label[0],  gallery_clabel[0])
        # print(gallery_name[10], gallery_label[10], gallery_clabel[10])
        # print(gallery_name[20], gallery_label[20], gallery_clabel[20])
        # print(gallery_name[30], gallery_label[30], gallery_clabel[30])

        #all vimage applied as probe/query
        query_img = test_rgb_image
        query_label = test_rgb_label
        query_clabel = test_rgb_clabel
        query_name = test_rgb_name


        print('gallery visible', len(gallery_label))
        print('query visible', len(query_label))
        # #shuffle
        # index = [i for i in range(len(test_rgb_name))]
        # np.random.shuffle(index)
        # query_img = query_img[index,:,:,:]
        # query_label = query_label[index]
        # query_clabel = query_clabel[index]
        # query_name_=[]
        # for i in range(len(test_rgb_name)):
        #     query_name_.append(query_name[index[i]])
        # print('shufule',query_img.shape,query_label.shape)
        #
        # #sample
        # sampleidx= [i for i in range(0, len(test_rgb_name), 10)]
        # query_img = query_img[sampleidx,:,:,:]
        # query_label = query_label[sampleidx]
        # query_clabel = query_clabel[sampleidx]
        # query_name__ = []
        # for i in range(len(sampleidx)):
        #     query_name__.append(query_name_[sampleidx[i]])
        #
        #
        # print('sample',query_img.shape,query_label.shape)
        #
        # print(query_name__[0],query_label[0],query_clabel[0])
        # print(query_name__[10], query_label[10], query_clabel[10])
        # print(query_name__[20], query_label[20], query_clabel[20])
        # print(query_name__[30], query_label[30], query_clabel[30])

        return query_img, query_label, query_clabel, gallery_img, gallery_label,gallery_clabel

    elif eval_mode=='t2v': #t2v
        # t--probe/query v---gallery
        print('.....................t2v...................')
        gallery_img = test_rgb_image
        gallery_label = test_rgb_label
        gallery_clabel = test_rgb_clabel
        gallery_name = test_rgb_name

        # all nir applied as probe/query
        query_img = test_nir_image
        query_label = test_nir_label
        query_clabel = test_nir_clabel
        query_name = test_nir_name


        print('gallery visible', len(gallery_label))

        print('query visible', len(query_label))
        # # shuffle
        # index = [i for i in range(len(test_nir_name))]
        # np.random.shuffle(index)
        # query_img = query_img[index, :, :, :]
        # query_label = query_label[index]
        # query_clabel = query_clabel[index]
        # query_name_=[]
        # for i in range(len(test_nir_name)):
        #     query_name_.append(query_name[index[i]])
        # print(query_img.shape, query_label.shape)
        #
        # # sample
        # sampleidx = [i for i in range(0, len(test_nir_name), 10)]
        # query_img = query_img[sampleidx, :, :, :]
        # query_label = query_label[sampleidx]
        # query_clabel = query_clabel[sampleidx]
        # query_name__ = []
        # for i in range(len(sampleidx)):
        #     query_name__.append(query_name_[index[i]])
        #
        # print(query_img.shape, query_label.shape)
        #
        #

        return query_img, query_label, query_clabel, gallery_img, gallery_label, gallery_clabel

    elif eval_mode=='mix': #t2v
        # t--probe/query v---gallery
        print('.....................mix..................')

        unilabel = np.unique(np.concatenate((test_rgb_label,test_nir_label),axis=0))

        gallery_idx_part1 = []
        gallery_idx_part2 = []
        query_idx_part1 = []
        query_idx_part2 = []
        for i in range(0,len(unilabel)):
            curlabel =  unilabel[i]
            rgbfullidx = np.arange(len(test_rgb_label))
            rgbidx = rgbfullidx[test_rgb_label==curlabel]
            for j in range(len(rgbidx)):
                if j % 2 == 0:
                    gallery_idx_part1.append(rgbidx[j])
                else:
                    query_idx_part1.append(rgbidx[j])

            nirfullidx = np.arange(len(test_nir_label))
            niridx = nirfullidx[test_nir_label==curlabel]
            for j in range(len(niridx)):
                if j % 2 == 0:
                    gallery_idx_part2.append(niridx[j])
                else:
                    query_idx_part2.append(niridx[j])


        gallery_idx_part1 = np.array(gallery_idx_part1)
        gallery_idx_part2 = np.array(gallery_idx_part2)
        query_idx_part1 = np.array(query_idx_part1)
        query_idx_part2 = np.array(query_idx_part2)


        gallery_img = np.concatenate((test_rgb_image[gallery_idx_part1,:,:,:],test_nir_image[gallery_idx_part2,:,:,:]),axis=0)
        gallery_label = np.concatenate((test_rgb_label[gallery_idx_part1],test_nir_label[gallery_idx_part2]),axis=0)#
        gallery_clabel = np.concatenate((test_rgb_clabel[gallery_idx_part1],test_nir_clabel[gallery_idx_part2]),axis=0)#

        gallery_name = []
        for i,v in enumerate(gallery_idx_part1):
            gallery_name.append(test_rgb_name[v])
        for i,v in enumerate(gallery_idx_part2):
            gallery_name.append(test_nir_name[v])


        # print(gallery_img.shape,gallery_label.shape,gallery_clabel.shape,len(gallery_name))

        # print(gallery_idx_part1(0),gallery_idx_part2.size(0))
        # for i in range(gallery_idx_part1.size(0)):
        #     gallery_name.append(test_rgb_name[gallery_idx_part1[i]])
        # for i in range(gallery_idx_part2.size(0)):
        #     gallery_name.append(test_nir_name[gallery_idx_part2[i]])


        query_img = np.concatenate((test_rgb_image[query_idx_part1,:,:,:],test_nir_image[query_idx_part2,:,:,:]),axis=0)
        query_label = np.concatenate((test_rgb_label[query_idx_part1],test_nir_label[query_idx_part2]),axis=0)#
        query_clabel = np.concatenate((test_rgb_clabel[query_idx_part1],test_nir_clabel[query_idx_part2]),axis=0)#


        query_name = []
        for i, v in enumerate(query_idx_part1):
            query_name.append(test_rgb_name[v])
        for i, v in enumerate(query_idx_part2):
            query_name.append(test_nir_name[v])

        # print(len(query_name))

        return query_img, query_label, query_clabel, gallery_img, gallery_label, gallery_clabel

def process_query_llcm(data_path, mode = 'v2t', relabel=False):
    if mode== 'v2t':
        print('query has rgb images')
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode =='t2v':
        print('query  has nir images')
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
    else:
        print('query mode wrong....')
        assert(1==0)
    
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, mode = 'v2t', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 't2v':
        print('gallery has rgb images')
        cameras = ['test_vis/cam1','test_vis/cam2','test_vis/cam3','test_vis/cam4','test_vis/cam5','test_vis/cam6','test_vis/cam7','test_vis/cam8','test_vis/cam9']
    elif mode =='v2t':
        print('gallery has nir images')
        cameras = ['test_nir/cam1','test_nir/cam2','test_nir/cam4','test_nir/cam5','test_nir/cam6','test_nir/cam7','test_nir/cam8','test_nir/cam9']
    else:
        print('gallery mode wrong...')
        assert(1==0)
        
    file_path = os.path.join(data_path,'idx/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    
def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_msvr310(img_dir, trial=-1, modal=None):

    if modal == 'vis':
        input_data_path = img_dir + '/trial{}'.format(trial) + '/tevis.txt'
        print('test:',input_data_path)
    elif modal == 'ni':
        input_data_path = img_dir + '/trial{}'.format(trial) + '/teni.txt'
        print('test:',input_data_path)





    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()

        file_label = []
        file_clabel = []
        file_image = []
        for i in range(len(data_file_list)):
            pos = str.find(data_file_list[i], '_')
            cur_pid = int(data_file_list[i][pos - 4:pos])
            file_label.append(cur_pid)

            pos = str.find(data_file_list[i], '_v')+2
            cur_clabel = data_file_list[i][pos]
            file_clabel.append(cur_clabel)

            img = Image.open(data_file_list[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            file_image.append(pix_array)

    file_image = np.array(file_image)

    return file_image, np.array(file_label), np.array(file_clabel)

def process_test_regdb(img_dir, trial = 1, modal = 'v'):
    if modal=='v':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='t':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)


def process_test_dn348(img_dir, modal=None, doquery=None):
    if modal == 'v2t':
        if doquery :
            input_data_path = img_dir  + 'train_test_split/test_list_day.txt'
            datapath_simi = img_dir + '/day/'
        else:
            input_data_path = img_dir + 'train_test_split/test_list_night.txt'
            datapath_simi = img_dir + '/night/'

    elif modal == 't2v':
        if doquery :
            input_data_path = img_dir + 'train_test_split/test_list_night.txt'
            datapath_simi = img_dir + '/night/'
        else:
            input_data_path = img_dir + 'train_test_split/test_list_day.txt'
            datapath_simi = img_dir + '/day/'

    img_dir = img_dir + modal
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[0]) for s in data_file_list]

        pid_container = set()

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i] = pid

    return file_image, np.array(file_label), datapath_simi


def process_test_dnwild(img_dir, modal=None, doquery=None):
    if modal == 'v2t':
        if doquery :
            input_data_path = img_dir  + 'train_test_split/day_test.txt'
            datapath_simi = img_dir + '/day/'
        else:
            input_data_path = img_dir + 'train_test_split/night_test.txt'
            datapath_simi = img_dir + '/night/'

    elif modal == 't2v':
        if doquery :
            input_data_path = img_dir + 'train_test_split/night_test.txt'
            datapath_simi = img_dir + '/night/'
        else:
            input_data_path = img_dir + 'train_test_split/day_test.txt'
            datapath_simi = img_dir + '/day/'


    img_dir = img_dir + modal
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split('/')[0]) for s in data_file_list]

        pid_container = set()

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i] = pid

    return file_image, np.array(file_label), datapath_simi
