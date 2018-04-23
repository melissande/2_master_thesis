import sys
import numpy as np
import os
import cv2
import logging
from image_utils import standardize,distance_map_batch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn
from dataset_generator_2 import Dataset_sat
from torch.utils.data import DataLoader
from IOU_computations import *
from random import randint
import time
from shapeSorter import SimpleSegmentationDataset



#####################################
GLOBAL_PATH='MODEL_BASIC_TOT_TEST_120/'


if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
######################################


########Which model ?###############
# from unet_val import UNet
from unet_val_2 import UNet
# from unet_meli import UNet,weights_init
WEIGHTS_INIT=False

####################################

INPUT_CHANNELS=9 #3 for Shape dataset and 9 for Sat dataset
NB_CLASSES=2 #3 for shape and 2 for sat dataset
SIZE_PATCH=120
##############
MODEL_PATH_SAVE=GLOBAL_PATH+'RESUNET_test_check_alright'
MODEL_PATH_RESTORE=''
TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

##############

REC_SAVE=2000
DROPOUT=0.2
DEFAULT_BATCH_SIZE = 8# 32 for spacenet and 8 for ghana
DEFAULT_EPOCHS =3
DEFAULT_VALID=100 #around 1200 elements in ghana validation and 15000 in spacenet validation
DISPLAY_STEP=100 #100
IOU_STEP=2

###############
DEFAULT_LAYERS=3
DEFAULT_FEATURES_ROOT=32
DEFAULT_FILTERS_SIZE=3
DEFAULT_LR=0.0001#0.0001
DEFAULT_FILTER_WIDTH=3

###Tune Learning rate
REDUCE_LR_STEPS = [1,5, 50, 100,200] #reduce of half everytime

################
DISTANCE_NET=False
BINS=15
THRESHOLD=33

####### TMP folder for IOU

TMP_IOU=TEST_SAVE+'TMP_IOU/'
if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)

#######  Data
#     root_folder ='/scratch/SPACENET_DATA_PROCESSED/DATASET/120_x_120_8_bands_pansh/'
root_folder = '../2_DATA_GHANA/DATASET/120_x_120_8_pansh/'
# root_folder=''




class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param lr: learning rate
    """
    def __init__(self, net, batch_size=10, lr=0.0001,nb_classes=2):
        self.net = net
        self.batch_size = batch_size
        self.lr = lr
        self.nb_classes=nb_classes
    def _initialize(self, prediction_path,avg_loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif,iou_step,dist_net,threshold,bins):
        
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        self.prediction_path = prediction_path
        self.avg_loss_train=avg_loss_train
        self.file_train=file_train
        self.loss_verif=loss_verif
        self.file_verif=file_verif
        self.IOU_verif=IOU_verif
        self.IOU_file_verif=IOU_file_verif
        self.IOU_acc_verif=IOU_acc_verif
        self.IOU_acc_file_verif=IOU_acc_file_verif
        self.f1_IOU_verif=f1_IOU_verif
        self.f1_IOU_file_verif=f1_IOU_file_verif
        self.IOU_STEP=iou_step
        self.threshold=threshold
        self.bins=bins
        self.dist_net=dist_net
        
    def train(self, data_provider_path, save_path='', restore_path='',  epochs=3, dropout=0.2, display_step=100, validation_batch_size=30,rec_save=2000, prediction_path = '',dist_net=False,threshold=20,bins=15,iou_step=1,reduce_lr_steps=[1,10,100,200],data_aug=None):
        """
        Lauches the training process
        
        :param data_provider_path: where the DATASET folder is
        :param save_path: path where to store checkpoints
        :param restore_path: path where is the model to restore is stored
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param prediction_path: path where to save predictions on each epoch
        """
        
        PATH_TRAINING=data_provider_path+'TRAINING/'
        PATH_VALIDATION=data_provider_path+'VALIDATION/'
        PATH_TEST=data_provider_path+'TEST/'
        
        
#         ###Visualize Training  loss and lr
#         fig,axs =plt.subplots(1, 3,figsize=(9,3))
#         draw_update([],[],[],fig,axs)
        
       
        
        if epochs == 0:
            return save_path
        if save_path=='':
            return 'Specify a path where to store the Model'
        
        if prediction_path=='':
            return 'Specify where to stored visualization of training'
            
        if restore_path=='':
            lr_train,loss_train,avg_loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(prediction_path,'w')
            print('Model trained from scratch')
        else:
            lr_train,loss_train,avg_loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(prediction_path,'a')
            self.net.load_state_dict(torch.load(restore_path))
            print('Model loaded from {}'.format(restore_path))
        
        TMP_IOU=prediction_path+'TMP_IOU/'
        if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)
        self._initialize(prediction_path,avg_loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif,iou_step,dist_net,threshold,bins)
         
            
        ###Validation loader
        if data_provider_path=='': ##For Shape
            val_generator=SimpleSegmentationDataset(SIZE_PATCH, 3, alpha =1.0,virtual_size=1200)#see tot val set in config
        else: ##for sat data
            val_generator=Dataset_sat.from_root_folder(PATH_VALIDATION,self.nb_classes)
        
        val_loader = DataLoader(val_generator, batch_size=validation_batch_size,shuffle=False, num_workers=1)
        RBD=randint(0,int(val_loader.__len__())-1)
        self.store_init_and_last(val_loader,"_init",RBD)
        
        ###Training loader
        if data_provider_path=='':##For Shape
            train_generator=SimpleSegmentationDataset(SIZE_PATCH, 3, alpha =1.0,virtual_size=4960)#4960
        else:
            train_generator=Dataset_sat.from_root_folder(PATH_TRAINING,self.nb_classes)
        
        train_loader = DataLoader(train_generator, batch_size=self.batch_size,shuffle=True, num_workers=1)
        
        
        logging.info("Start optimization")
        counter=0
        for epoch in range(epochs):
            
            ##tune learning reate
            if epoch in reduce_lr_steps:
                self.lr = self.lr * 0.5
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            
            total_loss = 0
            error_tot=0   
            train_loader = DataLoader(train_generator, batch_size=self.batch_size,shuffle=True, num_workers=1)
            for i_batch,sample_batch in enumerate(train_loader):
                batch_x=standardize(sample_batch['input'])
                batch_y=sample_batch['groundtruth']
                
                ##Variables input and output transformed for cuda
                X = Variable(batch_x.float())
                X=X.permute(0,3,1,2).cuda()  
                Y = Variable(batch_y.float())
                Y=Y.cuda()
                
                ## fwd
                if self.dist_net:
                    
                    Y_dist=distance_map_batch(batch_y,self.threshold,self.bins)
                    Y_dist = Variable(Y_dist.float())
                    Y_dist=Y_dist.cuda()
    
                    probs_dist,probs_seg=predict(self.net,X,self.dist_net)
                    self.optimizer.zero_grad()
                    loss_seg=criterion(Y,probs_seg)
                    loss_dist=criterion(Y_dist,probs_dist)
                    loss=loss_seg+loss_dist
                    
                else:
                    
                    probs=predict(self.net,X)
                    self.optimizer.zero_grad()
                    loss=criterion(Y,probs)
                    Y_dist=None
                    
                loss.backward()
                self.optimizer.step()
                
                total_loss+=loss.data[0]
                loss_train.append(loss.data[0])
                lr_train.append(self.lr)
                
                counter+=1
                
                if i_batch % display_step == 0:
                    self.output_training_stats(i_batch,X,Y,Y_dist)
#                     draw_update(loss_train,self.avg_loss_train,lr_train,fig,axs)
                    
                if counter % rec_save == 0:
                    torch.save(self.net.state_dict(),save_path + 'CP{}.pth'.format(counter))
                    print('Checkpoint {} saved !'.format(counter))
            
            self.avg_loss_train.append(total_loss/train_loader.__len__())
            (self.file_train).write(str(total_loss/train_loader.__len__())+'\n')
            logging.info(" Training {:}, Minibatch Loss= {:.4f}".format("epoch_%s"%epoch,total_loss/train_loader.__len__()))
            self.store_validation(val_loader,epoch,RBD,file_gson=TMP_IOU,save_patches=False)
            
    
        self.store_init_and_last(val_loader,'_last_',RBD,save_patches=True)
        time.sleep(4)
        plt.close(fig)
        return save_path, loss_train,self.avg_loss_train,self.loss_verif,self.IOU_verif,self.IOU_acc_verif,self.f1_IOU_verif      
        
    def output_training_stats(self, step, batch_x, batch_y,batch_y_dist=None):
        # Calculate batch loss and accuracy
        if self.dist_net:
            predictions_dist,predictions_seg=predict(self.net,batch_x,self.dist_net)
            loss_seg=criterion(batch_y,predictions_seg)
            loss_dist=criterion(batch_y_dist,predictions_dist)
            loss=loss_seg+loss_dist
            loss=loss.data[0]
            predictions=predictions_seg.data.cpu().numpy()
        else:
            predictions=predict(self.net,batch_x)
            loss=criterion(batch_y,predictions)
            loss=loss.data[0]
            predictions=predictions.data.cpu().numpy()
        
        groundtruth=batch_y.data.cpu().numpy()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Minibatch error= {:.4f}%".format(step,loss,error_rate(predictions, groundtruth)))
    
    def store_init_and_last(self,val_loader,name,random_batch_display,*,save_patches=True):
        loss_v=0
        error_rate_v=0
        
        for i_batch,sample in enumerate(val_loader):
            batch_x=standardize(sample['input'])
            batch_y=sample['groundtruth']
            
            X = Variable(batch_x.float())
            X=X.permute(0,3,1,2).cuda()  
            Y = Variable(batch_y.float())
            Y=Y.cuda()
            if self.dist_net:
                y_dist=distance_map_batch(batch_y,self.threshold,self.bins)
                Y_dist = Variable(y_dist.float())
                Y_dist=Y_dist.cuda()
                
                probs_dist,probs_seg=predict(self.net,X,self.dist_net)

                loss_seg=criterion(Y,probs_seg)

                loss_dist=criterion(Y_dist,probs_dist)
                loss=loss_seg+loss_dist
                
                prediction_seg_v=probs_seg.data.cpu().numpy()
                groundtruth_seg_v=np.asarray(batch_y)
                prediction_dist_v=probs_dist.data.cpu().numpy()
                groundtruth_dist=np.asarray(y_dist)

            
            else:
                probs=predict(self.net,X)
                loss=criterion(Y,probs)
                
                prediction_seg_v=probs.data.cpu().numpy()
                groundtruth_seg_v=np.asarray(batch_y)
                prediction_dist_v=None
                groundtruth_dist=None
                
            
            loss_v+=loss.data[0]
            
            error_rate_v+=error_rate(prediction_seg_v,groundtruth_seg_v)
            
            if i_batch==random_batch_display:
                batch_x=np.asarray(batch_x)
                if batch_x.shape[-1]>3:
                    pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)
                else:
                    pansharp=batch_x
                plot_summary(prediction_seg_v,groundtruth_seg_v,prediction_dist_v,groundtruth_dist,pansharp,name,self.prediction_path,save_patches)

    
                         
        loss_v/=val_loader.__len__()   
        error_rate_v/=val_loader.__len__()  
        logging.info("Verification  loss= {:.4f},error= {:.4f}%".format(loss_v,error_rate_v))
        
    def store_validation(self,val_loader, epoch,random_batch_display,*,file_gson='',save_patches=False):
        loss_v=0
        iou_v=0
        iou_acc_v=0
        f1_v=0
        error_rate_v=0
        name="epoch_%s"%epoch
        
        for i_batch,sample in enumerate(val_loader):
            batch_x=standardize(sample['input'])
            batch_y=sample['groundtruth']
            
            X = Variable(batch_x.float())
            X=X.permute(0,3,1,2).cuda()  
            Y = Variable(batch_y.float())
            Y=Y.cuda()
            if self.dist_net:
                y_dist=distance_map_batch(batch_y,self.threshold,self.bins)
                Y_dist = Variable(y_dist.float())
                Y_dist=Y_dist.cuda()
                probs_dist,probs_seg=predict(self.net,X,self.dist_net)
                loss_seg=criterion(Y,probs_seg)
                loss_dist=criterion(Y_dist,probs_dist)
                loss=loss_seg+loss_dist
                prediction_seg_v=probs_seg.data.cpu().numpy()
                groundtruth_seg_v=np.asarray(batch_y)
                prediction_dist_v=probs_dist.data.cpu().numpy()
                groundtruth_dist=np.asarray(y_dist)
                
            else:
                probs=predict(self.net,X)
                loss=criterion(Y,probs)
                prediction_seg_v=probs.data.cpu().numpy()
                groundtruth_seg_v=np.asarray(batch_y)
                prediction_dist_v=None
                groundtruth_dist=None
                
            loss_v+=loss.data[0]
            

            
            if (epoch+1)%self.IOU_STEP==0:
                iou_acc,f1,iou=predict_score_batch(file_gson,np.argmax(groundtruth_seg_v,3),np.argmax(prediction_seg_v,3))
                iou_acc_v+=iou_acc
                iou_v+=iou
                f1_v+=f1
                
            error_rate_v+=error_rate(prediction_seg_v,groundtruth_seg_v)
            if i_batch==random_batch_display:
                batch_x=np.asarray(batch_x)
                if batch_x.shape[-1]>3:
                    pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)
                else:
                    pansharp=batch_x
#                 plot_summary(prediction_seg_v,groundtruth_seg_v,prediction_dist_v,groundtruth_dist,pansharp,name,self.prediction_path,save_patches)


        loss_v/=val_loader.__len__()
        error_rate_v/=val_loader.__len__()
        self.loss_verif.append(loss_v)
        (self.file_verif).write(str(loss_v)+'\n')
        if (epoch+1)%self.IOU_STEP==0:
            iou_v/=val_loader.__len__()  
            iou_acc_v/=val_loader.__len__()  
            f1_v/=val_loader.__len__()  
            logging.info("Verification  IOU = {:.4f}, IOU Precision = {:.4f}%, F1 IOU= {:.4f}%".format(iou_v,iou_acc_v,f1_v))
            self.IOU_verif.append(iou_v)
            self.IOU_acc_verif.append(iou_acc_v)
            self.f1_IOU_verif.append(f1_v)
            
            
            (self.IOU_file_verif).write(str(iou_acc_v)+'\n')
            (self.IOU_acc_file_verif).write(str(iou_acc_v)+'\n')
            (self.f1_IOU_file_verif).write(str(f1_v)+'\n')
            

        logging.info("Verification  loss= {:.4f},error rate= {:.4f}%".format(loss_v,error_rate_v)) 

loss_fn=nn.CrossEntropyLoss()
def criterion(y,y_):
    
    y = y.contiguous().view(-1,y.size()[-1])
    y_ = y_.contiguous().view(-1,y.size()[-1])
    y = y.max(-1)[1]
    loss = loss_fn(y_,y)
    
    return loss


def predict(net,batch_x,dist_net=False):
    
    if dist_net:
        logits_dist,logits_seg=net(batch_x)
        return logits_dist.permute(0,2,3,1),logits_seg.permute(0,2,3,1)  
    else:
        logits=net(batch_x)
        return logits.permute(0,2,3,1)
    
def save_metrics(prediction_path,mode):
    #STORE loss for ANALYSIS
    lr_train=[]
    loss_train=[]
    avg_loss_train=[]
    file_train = open(prediction_path+'avg_loss_train.txt',mode) 
    loss_verif=[]
    file_verif = open(prediction_path+'loss_verif.txt',mode) 
    #STORE IOU for ANALYSIS
    IOU_verif=[]
    IOU_file_verif = open(prediction_path+'iou_verif.txt',mode)
    #STORE IOU_ACC for ANALYSIS
    IOU_acc_verif=[]
    IOU_acc_file_verif = open(prediction_path+'iou_acc_verif.txt',mode)
    #STORE f1_IOU for ANALYSIS
    f1_IOU_verif=[]
    f1_IOU_file_verif = open(prediction_path+'f1_iou_verif.txt',mode) 
    
    return lr_train,loss_train,avg_loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif
def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))
def plot_summary(predictions,labels,prediction_dist,groundtruth_dist,pansharp,epoch,prediction_path,save_patches):

    if prediction_dist is None and groundtruth_dist is None:
#         fig,axs=plt.subplots(3, len(pansharp),figsize=(3*len(pansharp),9))

        labels=np.argmax(labels, 3) 
        logits=np.argmax(predictions, 3)

        for i in range(len(pansharp)):

#             axs[0,i].imshow(pansharp[i])
#             axs[1,i].imshow(labels[i]) 
#             axs[2,i].imshow(logits[i])


            if save_patches:
                plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels[i])
                plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',logits[i])
    else:
        
#         fig,axs=plt.subplots(5, len(pansharp),figsize=(5*len(pansharp),15))

        labels_seg=np.argmax(labels, 3) 
        logits_seg=np.argmax(predictions, 3)
        labels_dist=np.argmax(groundtruth_dist, 3) 
        logits_dist=np.argmax(prediction_dist, 3)

        for i in range(len(pansharp)):

#             axs[0,i].imshow(pansharp[i])
#             axs[1,i].imshow(labels_seg[i]) 
#             axs[2,i].imshow(logits_seg[i])
#             axs[3,i].imshow(labels_dist[i],cmap="jet")
#             axs[4,i].imshow(logits_dist[i],cmap="jet")


            if save_patches:
                plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',pansharp[i])
                plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels_seg[i])
                plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',logits_seg[i])
                plt.imsave(prediction_path+epoch+'_Groundtruth_dist_'+str(i)+'.jpg',labels_dist[i],cmap="jet")
                plt.imsave(prediction_path+epoch+'_Predictions_dist_'+str(i)+'.jpg',logits_dist[i],cmap="jet")

#     fig.canvas.draw()
#     time.sleep(4)
#     plt.close(fig)
    
# def draw_update(loss,avg_loss_train,lr,fig,axs):
    
#     if len(loss)==0 and len(lr)==0 and len(avg_loss_train)==0:

#         axs[0].set_ylim(0,0.001)
#         axs[0].set_title('Loss train')
#         axs[1].set_ylim(0,0.001)
#         axs[1].set_title('Avg loss train')
#         axs[2].set_ylim(0,1)
#         axs[2].set_title('Learning rate')

#     else:
#         global_step=len(loss)
#         epoch=np.arange(len(avg_loss_train))
#         ite=np.arange(global_step)
#         axs[0].clear()
#         axs[1].clear()
#         axs[2].clear()
#         line1, = axs[0].plot(ite, loss, 'r-') 
#         line1, = axs[1].plot(epoch, avg_loss_train, 'g-') 
#         line2, = axs[2].plot(ite, lr, 'b-') 
#         fig.canvas.draw()
#         time.sleep(1)
 

if __name__ == '__main__':

#     python resunet_main_fin.py ../2_DATA_GHANA/DATASET/120_x_120_8_pansh/ MODEL_GHANA_TEST/ RESUNET_ghana_test '' --input_channels=9 --nb_classes=2  --learning_rate=1e-3 --batch_size=8  --epochs=3 --display_step=100 --rec_save_model=2000 --distance_net=False --iou_step=15 --lr_reduce_steps=1,5,50,100,200

# python resunet_main_fin.py /scratch/SPACENET_DATA_PROCESSED/DATASET/120_x_120_8_bands_pansh/ MODEL_SPACENET_NODIST_true/ RESUNET_spacenet_nodist '' --input_channels=9 --nb_classes=2  --learning_rate=1e-3 --batch_size=32  --epochs=250 --display_step=1000 --rec_save_model=10000 --distance_net=False --iou_step=15 --lr_reduce_steps=1,5,50,100,200

# python resunet_main_fin.py '' MODEL_SHAPE_DIST/ RESUNET_shape_test '' --input_channels=3 --nb_classes=2  --learning_rate=1e-2 --batch_size=32  --epochs=80 --display_step=100 --rec_save_model=2000 --distance_net=True --iou_step=10 --lr_reduce_steps=1,5,10,50,70
    
    
    root_folder=sys.argv[1]
     ##########
    GLOBAL_PATH=sys.argv[2]
    

    if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
    TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
    if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
    ##########
    
    
    MODEL_PATH_SAVE=GLOBAL_PATH+sys.argv[3]
    MODEL_PATH_RESTORE=sys.argv[4]
    
    for i in range(5, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--input_channels'):
            INPUT_CHANNELS=int(arg[len('--input_channels='):])
        elif arg.startswith('--nb_classes'):
            NB_CLASSES=int(arg[len('--nb_classes='):])
        elif arg.startswith('--nb_layers'):
            DEFAULT_LAYERS=int(arg[len('--nb_layers='):])
        elif arg.startswith('--nb_features_root'):
            DEFAULT_FEATURES_ROOT=int(arg[len('--nb_features_root='):])
        elif arg.startswith('--learning_rate'):
            DEFAULT_LR=float(arg[len('--learning_rate='):])
        elif arg.startswith('--batch_size'):
            DEFAULT_BATCH_SIZE = int(arg[len('--batch_size='):])
        elif arg.startswith('--epochs'):
            DEFAULT_EPOCHS = int(arg[len('--epochs='):])
        elif arg.startswith('--dropout'):
            DROPOUT = float(arg[len('--dropout='):])
        elif arg.startswith('--display_step'):
            DISPLAY_STEP = int(arg[len('--display_step='):])
        elif arg.startswith('--validation_size_batch'):
            DEFAULT_VALID = int(arg[len('--validation_size_batch='):])  
        elif arg.startswith('--rec_save_model'):
            REC_SAVE = int(arg[len('--rec_save_model='):])
        elif arg.startswith('--distance_net'):
            DISTANCE_NET = arg[len('--distance_net='):]
        elif arg.startswith('--iou_step'):
            IOU_STEP = int(arg[len('--iou_step='):])
        elif arg.startswith('--lr_reduce_steps'):
            REDUCE_LR_STEPS = np.asarray(arg[len('--lr_reduce_steps='):].split(',')).astype(int)
            
        else:
            raise ValueError('Unknown argument %s' % str(arg))
            
            
    model=UNet(INPUT_CHANNELS,NB_CLASSES,depth =DEFAULT_LAYERS,n_features_zero =DEFAULT_FEATURES_ROOT,width_kernel=DEFAULT_FILTER_WIDTH,dropout=DROPOUT,distance_net=DISTANCE_NET,bins=BINS)
    
    if WEIGHTS_INIT:
        model.apply(weights_init)

    model.cuda()
    
    cudnn.benchmark = True

    
    trainer=Trainer(model,DEFAULT_BATCH_SIZE,DEFAULT_LR,NB_CLASSES)
    save_path,loss_train,avg_loss_train,loss_verif,iou_verif,iou_acc_verif,f1_iou_verif=trainer.train( root_folder, MODEL_PATH_SAVE, MODEL_PATH_RESTORE,DEFAULT_EPOCHS,DROPOUT, DISPLAY_STEP, DEFAULT_VALID,REC_SAVE, TEST_SAVE,DISTANCE_NET,THRESHOLD,BINS,IOU_STEP,REDUCE_LR_STEPS)
    print('Last model saved is %s: '%save_path)
    
#     fig, axs = plt.subplots(5, sharex=True)

    
#     #SAVE loss

#     axs[0].set_title('Plot Loss', fontsize=8)
#     ite = np.arange(0,len(loss_train),1)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     axs[0].plot(ite,loss_train,'b',epo,loss_verif,'g',epo,avg_loss_train,'r')
#     axs[0].set_ylabel('Loss')

    
#      #SAVE IOU

#     axs[1].set_title('Plot IOU', fontsize=8)
#     epo=np.arange(int(len(loss_train)/len(iou_verif))-1,len(loss_train),int(len(loss_train)/len(iou_verif)))
#     axs[1].plot(epo,iou_verif,'g')
#     axs[1].set_ylabel('IOU in %')

    
#     #SAVE IOU  acc

#     axs[2].set_title('Plot IOU Accuracy', fontsize=8)
#     epo=np.arange(int(len(loss_train)/len(iou_acc_verif))-1,len(loss_train),int(len(loss_train)/len(iou_acc_verif)))
#     axs[2].plot(epo,iou_acc_verif,'g')
#     axs[2].set_ylabel('IOU Accuracy in %')
    


    
#      #SAVE f1 IOU

#     axs[3].set_title('Plot f1 IOU', fontsize=8)
#     epo=np.arange(int(len(loss_train)/len(f1_iou_verif))-1,len(loss_train),int(len(loss_train)/len(f1_iou_verif)))
#     axs[3].plot(epo,f1_iou_verif,'g')
#     axs[3].set_ylabel('f1 IOU in %')
    
    
#      #SAVE loss 2

#     axs[4].set_title('Plot Loss', fontsize=8)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     axs[4].plot(epo,loss_verif,'g',epo,avg_loss_train,'r')
#     axs[4].set_ylabel('Loss')

    
#     fig.canvas.draw()
#     time.sleep(1)
#     plt.close(fig)

        
