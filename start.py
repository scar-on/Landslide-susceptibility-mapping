import read_data
from tensorboardX import SummaryWriter
import config
import generate_LSM
#实例化config
config=config.config
import train_cnn

is_train = True
is_saveLSM = True

if __name__ == '__main__':
    if is_train:
        print('***************************************读取训练集 测试集***************************************')
        alldata_train,alltarget_train = read_data.train_data()
        alldata_val,alltarget_val = read_data.test_data()
        print('*******************************************开始训练*******************************************')
        train_cnn.train(alldata_train, alltarget_train, alldata_val, alltarget_val)
    if is_saveLSM:
        generate_LSM.save_LSM()