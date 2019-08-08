import os

class WiderFaceLoader():
    """Wider人脸检测数据集读取类
    返回：图片文件路径，人脸框坐标
    """
    def __init__(self):
        pass

    def load(self,file_path,img_dir):
        train_data=[]
        train_label=[]
        with open(file_path, 'r', encoding='UTF-8') as txt_file:
            lines = txt_file.read().split('\n')
            line_index = 0
            while True:
                # 读取图片名
                img_name = lines[line_index]
                line_index+=1
                # print('读取文件：', img_name)
                if img_name=='':
                    break
                # 读取目标框数量
                img_num = int(lines[line_index])
                line_index+=1
                # 循环读取目标框数据
                boxs=[]
                for _ in range(img_num):
                    img_box = lines[line_index].split(' ')
                    line_index+=1
                    boxs.append([float(img_box[0]),float(img_box[1]),float(img_box[2]),float(img_box[3])])
                if len(boxs)>0:
                    train_data.append(os.path.join(img_dir, img_name))
                    train_label.append(boxs)
                else:
                    # 无人脸框的跳一行
                    line_index+=1
        return train_data, train_label


def main():
    wider_face_loader = WiderFaceLoader()
    train_data, train_label=wider_face_loader.load(u'E:\MyFiles\人脸检测素材\wider_face_split\wider_face_train_bbx_gt.txt',u'E:\MyFiles\人脸检测素材\WIDER_train\images')
    print('train_data',train_data[2])
    print('train_label',train_label[2])


if __name__ == '__main__':
    main()