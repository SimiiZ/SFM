import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import os

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RGBN300Data(data.Dataset):
    def __init__(self, data_dir, transform_rgb=None, transform_gray=None, transform_nir=None, rgbIndex=None,
                 niIndex=None, imgh=None, imgw=None):

        train_rgb_path = data_dir + '/R/'
        train_ni_path = data_dir + '/N/'
        # train_gray_path = data_dir + '/Gray/'
        print(imgh, imgw)

        train_rgb_list_all = os.listdir(train_rgb_path)  # all id in the rgb part of the whole dataset
        # train_gray_list_all = os.listdir(train_gray_path)  # all id in the rgb part of the whole dataset
        train_nir_list_all = os.listdir(train_ni_path)  # all id in the nir part of the whole dataset


        #############################select odd to construct training set#################
        # print(train_rgb_list, train_ni_list)
        train_rgb_list = []  # select odd
        for i in range(len(train_rgb_list_all)):
            cur_id = train_rgb_list_all[i]
            cur_id = int(cur_id)  # str2num
            if cur_id % 2 != 0:
                # print(cur_id)
                train_rgb_list.append(train_rgb_list_all[i])



        # print(len(train_rgb_list))
        train_nir_list = []  # select odd
        for i in range(len(train_nir_list_all)):
            cur_id = train_nir_list_all[i]
            cur_id = int(cur_id)  # str2num
            if cur_id % 2 != 0:
                # print(cur_id)
                train_nir_list.append(train_nir_list_all[i])

        train_rgb_name = []
        train_rgb_image = []
        for i in range(len(train_rgb_list)):
            cur_id_path = train_rgb_path + train_rgb_list[i]
            cur_id_list = os.listdir(cur_id_path)
            # print(cur_id_list)
            for j in range(len(cur_id_list)):
                imgname = cur_id_path + '/' + cur_id_list[j]
                if os.path.splitext(imgname)[1] == '.jpg':
                    img = Image.open(imgname)
                    img = img.resize((imgw, imgh), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    train_rgb_image.append(pix_array)
                    train_rgb_name.append(cur_id_list[j])
        train_rgb_image = np.array(train_rgb_image)



        train_nir_name = []
        train_nir_image = []
        for i in range(len(train_nir_list)):
            cur_id_path = train_ni_path + train_nir_list[i]
            cur_id_list = os.listdir(cur_id_path)
            for j in range(len(cur_id_list)):
                imgname = cur_id_path + '/' + cur_id_list[j]
                # print(imgname)
                if os.path.splitext(imgname)[1] == '.jpg':
                    img = Image.open(imgname)
                    img = img.resize((imgw, imgh), Image.ANTIALIAS)
                    pix_array = np.array(img)
                    train_nir_image.append(pix_array)
                    train_nir_name.append(cur_id_list[j])
        train_nir_image = np.array(train_nir_image)

        ################################process person id: tranform 0599 to 149.0##########################
        train_rgb_id = [s.split('_')[0] for s in train_rgb_name]
        # train_gray_id = [s.split('_')[0] for s in train_gray_name]
        train_nir_id = [s.split('_')[0] for s in train_nir_name]
        train_multi_modal_id = train_rgb_id + train_nir_id
        unique_id = np.unique(train_multi_modal_id)
        train_color_label = np.ones(len(train_rgb_id)) * -1
        # train_gray_label = np.ones(len(train_gray_id)) * -1
        train_thermal_label = np.ones(len(train_nir_id)) * -1

        for i in range(len(unique_id)):
            tmp_rgb = [k for k, v in enumerate(train_rgb_id) if v == unique_id[i]]
            train_color_label[tmp_rgb] = i

            tmp_nir = [k for k, v in enumerate(train_nir_id) if v == unique_id[i]]
            train_thermal_label[tmp_nir] = i

        ################################process camera id: tranform c0001 to 0.0 ##########################
        train_rgb_cid = [s.split('_')[1][1:] for s in train_rgb_name]
        # train_gray_cid = [s.split('_')[1][1:] for s in train_gray_name]
        train_nir_cid = [s.split('_')[1][1:] for s in train_nir_name]
        train_multi_modal_cid = train_rgb_cid + train_nir_cid
        unique_cid = np.unique(train_multi_modal_cid)
        train_rgb_clabel = np.ones(len(train_rgb_cid)) * -2
        # train_gray_clabel = np.ones(len(train_gray_cid)) * -2
        train_nir_clabel = np.ones(len(train_nir_cid)) * -2
        for i in range(len(unique_cid)):
            tmp_rgb = [k for k, v in enumerate(train_rgb_cid) if v == unique_cid[i]]
            train_rgb_clabel[tmp_rgb] = i


            tmp_nir = [k for k, v in enumerate(train_nir_cid) if v == unique_cid[i]]
            train_nir_clabel[tmp_nir] = i

        self.train_rgb_image = train_rgb_image
        self.train_color_label = train_color_label
        self.train_rgb_clabel = train_rgb_clabel
        self.train_rgb_name = train_rgb_name



        self.train_nir_image = train_nir_image
        self.train_thermal_label = train_thermal_label
        self.train_nir_clabel = train_nir_clabel
        self.train_nir_name = train_nir_name

        self.transform_rgb = transform_rgb
        self.transform_nir = transform_nir

        self.cIndex = rgbIndex
        self.tIndex = niIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_rgb_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_nir_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]


        # print(img1.shape, img2.shape)
        img1 = self.transform_rgb(img1)
        img2 = self.transform_nir(img2)

        return img1, img2, target1, target2


    def __len__(self):
        # print(len(self.train_color_label))
        return len(self.train_color_label)
    
class rgbn300_TestData(data.Dataset):
    def __init__(self, test_img, test_label, transform=None, img_size=(None, None)):
        self.test_image = test_img
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        #0613 change to long
        # 0613 change to long
        if isinstance(target1, np.ndarray):
            target1 = torch.from_numpy(target1).long()
        else:
            target1 = torch.tensor(target1).long()

        if isinstance(target2, np.ndarray):
            target2 = torch.from_numpy(target2).long()
        else:
            target2 = torch.tensor(target2).long()

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class LLCMData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, img_size = (144,288)):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        # self.transform_occ = transform_occ
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_normal = self.transform(img1)
        img2_normal = self.transform(img2)

        # img1_occ = self.transform_occ(img1)
        # img2_occ = self.transform_occ(img2)

        return img1_normal, img2_normal, target1, target2 #img1_occ, img2_occ

    def __len__(self):
        return len(self.train_color_label)

class LLCMData_dart(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, img_size = (144,288)):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            #print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        



class MSVR310Data(data.Dataset):
    def __init__(self, data_dir, trial, transform_rgb=None, transform_the=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        data_dir = '/datasets/msvr310/'
        train_color_list = data_dir + '/trial' + str(trial) + '/trvis.txt'
        train_thermal_list = data_dir +'/trial' + str(trial) + '/trni.txt'
        print('train:', train_color_list)
        print('train:', train_thermal_list)

        color_img_file, train_color_label = load_data_msvr310(train_color_list)
        thermal_img_file, train_thermal_label = load_data_msvr310(train_thermal_list)

        train_color_image = []
        train_color_clabel = []
        for i in range(len(color_img_file)):
            pos = str.find(color_img_file[i], '_v')+2
            cur_clabel = color_img_file[i][pos]
            # print(cur_clabel)
            train_color_clabel.append(cur_clabel)

            img = Image.open(color_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)
        train_color_clabel =  np.array(train_color_clabel)

        train_thermal_image = []
        train_thermal_clabel = []
        for i in range(len(thermal_img_file)):
            pos = str.find(thermal_img_file[i], '_v')+2
            cur_clabel = thermal_img_file[i][pos]
            # print(cur_clabel)
            train_thermal_clabel.append(cur_clabel)

            img = Image.open(thermal_img_file[i])
            img = img.resize((256, 256), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        train_thermal_clabel = np.array(train_thermal_clabel)










#---------------------------------------------------------------------------
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label
        self.train_color_clabel = train_color_clabel
        self.train_color_name = color_img_file


        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        self.train_thermal_clabel = train_thermal_clabel
        self.train_thermal_name = thermal_img_file

        self.transform_rgb = transform_rgb
        self.transform_the = transform_the








    def __getitem__(self, index):

        # print(index)
        rgb_idx = index[0]
        the_idx = index[1]
        #
        img1, target1 = self.train_color_image[rgb_idx], self.train_color_label[rgb_idx]
        img2, target2 = self.train_thermal_image[the_idx], self.train_thermal_label[the_idx]

        img1 = self.transform_rgb(img1)

        img2 = self.transform_the(img2)

        name1 = self.train_color_name[rgb_idx]
        name2 = self.train_thermal_name[the_idx]
        return img1, img2, target1, target2, name1, name2

    def __len__(self):
        # print(len(self.train_color_label))
        return len(self.train_color_label)

def load_data_msvr310(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = data_file_list
        file_label = []
        for i in range(len(file_image)):
            pos = str.find(file_image[i], '_')
            cur_pid = int(file_image[i][pos - 4:pos])
            file_label.append(cur_pid)
        # print(file_label)

    # #
    unipids = np.unique(file_label)
    file_label_continue = file_label.copy()
    for i in range(unipids.size):
        cur_pid = unipids[i]
        for j in range(len(file_label)):
            if cur_pid==file_label[j]:
                file_label_continue[j]=i


    # print('------',file_label_continue)

    # for j in range(len(file_label)):
    #     print(file_label[j],file_label_continue[j])


    return file_image, file_label_continue

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        self.test_img_file = test_img_file

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

class TestMSVR310Data(data.Dataset):
    def __init__(self, test_img, test_label, transform=None, img_size=(144, 288)):
        self.test_image = test_img
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


class DN348Data(data.Dataset):  # new
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None,img_size = (256,256)):
        # Load training images (path) and labels
        train_day_list = data_dir + 'train_test_split/train_list_day.txt'
        train_night_list = data_dir + 'train_test_split/train_list_night.txt'
        train_day_path = data_dir + 'day/'
        train_night_path = data_dir + 'night/'

        day_img_file, train_day_label = load_data1(train_day_list, train_day_path)
        night_img_file, train_night_label = load_data1(train_night_list, train_night_path)

        train_day_image = []
        for i in range(len(day_img_file)):
            img = Image.open(train_day_path + day_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_day_image.append(pix_array)
        train_day_image = np.array(train_day_image)

        train_night_image = []
        for i in range(len(night_img_file)):
            img = Image.open(train_night_path + night_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_night_image.append(pix_array)
            # print(pix_array.shape)
        train_night_image = np.array(train_night_image)

        # BGR to RGB
        self.train_color_image = train_day_image
        self.train_color_label = train_day_label

        # BGR to RGB
        self.train_thermal_image = train_night_image
        self.train_thermal_label = train_night_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_day_label)


class DNwildData(data.Dataset):  # new
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None,img_size = (256,256)):
        # Load training images (path) and labels
        train_day_list = data_dir + 'train_test_split/day_train.txt'
        train_night_list = data_dir + 'train_test_split/night_train.txt'
        train_day_path = data_dir + 'day/'
        train_night_path = data_dir + 'night/'

        day_img_file, train_day_label = load_data1(train_day_list, train_day_path)
        night_img_file, train_night_label = load_data1(train_night_list, train_night_path)

        train_day_image = []
        for i in range(len(day_img_file)):
            img = Image.open(train_day_path + day_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_day_image.append(pix_array)
        train_day_image = np.array(train_day_image)

        train_night_image = []
        for i in range(len(night_img_file)):
            img = Image.open(train_night_path + night_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_night_image.append(pix_array)
            # print(pix_array.shape)
        train_night_image = np.array(train_night_image)

        # BGR to RGB
        self.train_color_image = train_day_image
        self.train_color_label = train_day_label

        # BGR to RGB
        self.train_thermal_image = train_night_image
        self.train_thermal_label = train_night_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_day_label)


class TestDataOld1(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(256, 256)):
        test_image = []

        for i in range(len(test_img_file)):
            img = Image.open( data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def load_data1(input_data_path, dn_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]

        file_label = [int(s.split('/')[0]) for s in data_file_list]
        pid_container = set()

        for i in range(len(file_label)):
            # str1 = '%s/%s.jpg'%(self.all_dir,data_mat[i][0])
            # train_paths.append(img_paths[img_paths.index(str1)])
            pid = int(file_label[i])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for i in range(len(file_label)):
            pid = int(file_label[i])
            pid = pid2label[pid]
            file_label[i] = pid

    return file_image, file_label