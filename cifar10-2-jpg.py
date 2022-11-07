import numpy as np
# from scipy.misc import imsave
# import matplotlib.pyplot as plt
import os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

filename = './CIFAR10/cifar-10-batches-py'

meta = unpickle(filename+'/batches.meta')
label_name = meta[b'label_names']


for i in range(1, 6):
	content = unpickle(filename+'/data_batch_'+str(i))
	print('load data...')
	print(content.keys())
	print('tranfering data_batch' + str(i))
	for j in range(10000):
		img = content[b'data'][j]
		img = img.reshape(3,32,32)
		img = img.transpose(1,2,0)
		# imsave(img_name,img)
		dir_path = 'train/'+label_name[content[b'labels'][j]].decode() 
		img_name = dir_path + '/batch_' + str(i) + '_num_' + str(j) +'.jpg'

		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
   
		# plt.savefig(img_name+'.png')# 图像保存
		im = Image.fromarray(img)
		im.save(img_name+'.png') # 图像保存