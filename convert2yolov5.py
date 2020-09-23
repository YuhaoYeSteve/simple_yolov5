import yaml
import os
import shutil
import numpy as np

from ccutils import findFiles


def load_yaml(yaml_file):
	file = open(yaml_file, 'r', encoding="utf-8")
	file_data = file.read()
	file.close()
	data = yaml.load(file_data)
	return data



def init_label_map(names):
    label_map = {}
    for label, name in enumerate(names):
        label_map[name] = label
    return label_map


def load_txt_info(path):
    with open(path, 'r') as f:
        num = int(f.readline())
        width, height = f.readline().strip('\n').split(',')
        width, height = float(width), float(height)      
        infos = []
        for line in f.readlines():
            line = line.strip('\n').split(',')
            type = line[0]
            if type == 'point':
                continue
            if len(line) != 11:
                return [], False
            label_name = line[1]
            x1, y1, x2, y2 = float(line[2]), float(line[3]), float(line[5]), float(line[6])
            xmin, ymin, xmax, ymax = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)            
            
            x_c, y_c, w, h =  (x1 + x2) / 2, (y1 + y2) / 2, abs(x1-x2), abs(y1-y2)
            x_c, y_c, w, h =  x_c / width, y_c / height, w / width, h / height
            x_c, y_c = min(x_c, 1), min(y_c, 1)
            w, h = min(w, 1), min(h, 1)
            if label_name not in label_map.keys():
                continue
            label = label_map[label_name]
            info = np.array([label, x_c, y_c, w, h]).astype(str)      
            info = ' '.join(info)
            infos.append(info)
    return infos, True


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def convertor(path, convertor_img_dir, convertor_label_dir, filter):
    files = findFiles(path, "jpg")
    print('原始图片数量: ', len(files))

    for img_path in files:
        if filter not in img_path:
            continue
        txt_path = img_path.replace('.jpg', '.txt')
        if not os.path.exists(txt_path):
            print(f'txt: {txt_path} 不存在')
            continue

        infos, valid = load_txt_info(txt_path)

        if not valid:
            print(f'txt: {txt_path} 无效')
            continue


        convertor_label_path = os.path.join(convertor_label_dir, os.path.basename(txt_path))
        
        #print(convertor_label_path)
        with open(convertor_label_path, 'w') as f:
            for info in infos:
                f.write(info+'\n')

        shutil.copy(img_path, convertor_img_dir)


# load config
config = load_yaml('config.yaml')


# path
train_path, val_path = config['train_path'], config['val_path']
convertor_path = config['convertor_path']
names = config['names']

if os.path.exists(convertor_path):
    del_file(convertor_path)

train_image_path = os.path.join(convertor_path, 'images', 'train2017')
val_image_path = os.path.join(convertor_path, 'images', 'val2017')
train_label_path = os.path.join(convertor_path, 'labels', 'train2017')
val_label_path = os.path.join(convertor_path, 'labels', 'val2017')


if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)
if not os.path.exists(val_image_path):
    os.makedirs(val_image_path)
if not os.path.exists(train_label_path):
    os.makedirs(train_label_path)
if not os.path.exists(val_label_path):
    os.makedirs(val_label_path)






label_map = init_label_map(names)

convertor(train_path, train_image_path, train_label_path, config["filter"])
print('训练集转换后图片数量: ', len(os.listdir(train_image_path)))
convertor(val_path, val_image_path, val_label_path, config["filter"])
print('测集转换后图片数量: ', len(os.listdir(val_image_path)))



print('convertor finish')

















