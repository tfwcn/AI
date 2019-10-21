import os

class ImageLoader():
    """加载图片文件
    返回：图片文件路径
    """
    def __init__(self):
        pass

    def load(self,img_dir,sub_path=True):
        # 图片文件列表
        image_files = []
        # 查询子路径
        if sub_path:
            for dirpath, dirnames, filenames in os.walk(img_dir):
                for f in filenames:
                    # 转小写
                    fn = f.lower()
                    file_path=os.path.join(dirpath, f)
                    if fn[-4:]=='.png' or fn[-4:]=='.jpg' or fn[-4:]=='.jpeg':
                        image_files.append(file_path)
        else:
            # 只查询当前目录文件
            for file_or_dir_name in os.listdir(img_dir):
                dirpath=img_dir
                full_path=os.path.join(dirpath, file_or_dir_name)
                print(full_path)
                if os.path.isfile(file_or_dir_name):
                    f = full_path
                    # 转小写
                    fn = f.lower()
                    if fn[-4:]=='.png' or fn[-4:]=='.jpg' or fn[-4:]=='.jpeg':
                        file_path=os.path.join(dirpath, f)
                        image_files.append(file_path)
        return image_files


def main():
    image_loader = ImageLoader()
    image_files =image_loader.load(u'E:\MyFiles\人脸检测素材\WIDER_train\images')
    print('image_files',image_files[2])


if __name__ == '__main__':
    main()