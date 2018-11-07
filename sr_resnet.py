import os, sys
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from tqdm import tqdm

def sr_resnet(img_data):
    model = torch.load("./model_epoch_500.pth")["model"]
    model = model.cuda()
    height, width = img_data[0][0][0].shape[:2]

    with tqdm(total=len(img_data)) as pbar_p:
        for person in img_data:
            with tqdm(total=len(person)) as pbar_c:
                for i,imgs in enumerate(person):
                    im_inputL = imgs[0].astype(np.float32).transpose(2,0,1)
                    im_inputL = im_inputL.reshape(1,im_inputL.shape[0],
                            im_inputL.shape[1],
                            im_inputL.shape[2]
                            )
                    im_inputL = Variable(torch.from_numpy(im_inputL/255.).float())
                    im_inputR = imgs[1].astype(np.float32).transpose(2,0,1)
                    im_inputR = im_inputR.reshape(1,im_inputR.shape[0],
                            im_inputR.shape[1],
                            im_inputR.shape[2]
                            )
                    im_inputR = Variable(torch.from_numpy(im_inputR/255.).float())
                    im_inputL = im_inputL.cuda()
                    im_inputR = im_inputR.cuda()

                    outL = model(im_inputL).cpu().data[0].numpy().astype(np.float32)
                    outR = model(im_inputR).cpu().data[0].numpy().astype(np.float32)

                    outL = outL*255.
                    outL[outL<0] = 0
                    outL[outL>255.] = 255.
                    outL = outL.transpose(1,2,0)

                    outR = outR*255.
                    outR[outR<0] = 0
                    outR[outR>255.] = 255.
                    outR = outR.transpose(1,2,0)

                    outL = cv2.resize(outL, (width*4, height*4))
                    outR = cv2.resize(outR, (width*4, height*4))

                    person[i][0] = outL.astype(np.uint8)
                    person[i][1] = outR.astype(np.uint8)
                    pbar_c.update()
            pbar_p.update()
