import os


def findFiles(folder_path, filter, find_directory=False, include_sub_directory=True):
    find_result = []
    filter_list = []
    floder_root_count = 0

    if len(filter)>0:
        filter_list = filter.split(";")

    for root, dirs, files in os.walk(folder_path, topdown=True):
        if find_directory == True:
            for name in dirs:
                find_result.append(os.path.join(root, name))
            if(include_sub_directory == False):
                return  find_result
        else:
            if include_sub_directory == False:
                if floder_root_count>0: #如果不找下层目录，则只管第一级目录
                    continue

            for name in files:
                file_path = os.path.join(root, name)
                for type in filter_list:
                    if file_path.endswith(type):
                        find_result.append(file_path)
                        break

            floder_root_count +=1
            
    return find_result


def fileName(file_path, include_suffix=False):
    split_result = file_path
    if file_path.find('\\') !=-1:
        file_path = file_path.replace("\\", "/")

    if file_path.find('/') !=-1:
        split_result = file_path.split('/')[-1]

    if include_suffix ==False:
        if file_path.find('.') != -1:
            split_result = split_result.split('.')[0]
    return split_result

def mkdirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



if __name__ == '__main__':
    folder_path = '数据汇总'

    folders = findFiles(folder_path, 'jpg')
    for folder in folders:
        print(folder)
        








