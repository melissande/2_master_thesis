import sys
import numpy as np
import os
import cv2
import logging
from image_utils import standardize
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fu
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
# from unet_val import UNet
from unet_meli import UNet,weights_init
import torch.backends.cudnn as cudnn
from dataset_generator_2 import Dataset_sat
from torch.utils.data import DataLoader
from IOU_computations import *
from random import randint

##########
GLOBAL_PATH='MODEL_BASIC_TEST_120/'
##########

if not os.path.exists(GLOBAL_PATH):
            os.makedirs(GLOBAL_PATH)
        
#############
PATH_TRAINING='TRAINING/'
PATH_VALIDATION='VALIDATION/'
PATH_TEST='TEST/'

PATH_INPUT='INPUT/'
PATH_OUTPUT='OUTPUT/'
##############

        
INPUT_CHANNELS=9
NB_CLASSES=2

SIZE_PATCH=120
##############
MODEL_PATH_SAVE=GLOBAL_PATH+'RESUNET_pytorch_test'
MODEL_PATH_RESTORE=''
TEST_SAVE=GLOBAL_PATH+'TEST_SAVE/'
if not os.path.exists(TEST_SAVE):
            os.makedirs(TEST_SAVE)
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

##############

REC_SAVE=200#2000
DROPOUT=0.1#0.1
DEFAULT_BATCH_SIZE = 8#10
DEFAULT_EPOCHS = 1#50
DEFAULT_VALID=50#100
DISPLAY_STEP=100#50

###############
DEFAULT_LAYERS=3
DEFAULT_FEATURES_ROOT=32
DEFAULT_FILTERS_SIZE=3
DEFAULT_LR=0.0001

####### TMP folder for IOU

TMP_IOU=TEST_SAVE+'TMP_IOU/'
if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)
            
            
            
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
    def _initialize(self, prediction_path):
        
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        self.prediction_path = prediction_path
        
    
    def train(self, data_provider_path, save_path='', restore_path='',  epochs=3, dropout=0.1, display_step=1, validation_batch_size=30,rec_save=1, prediction_path = '',data_aug=None):
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
        
        
        ###Tune Learning rate
#         reduce_lr_steps = [1,5, 10, 100,200]
        reduce_lr_steps=[50,70]
        if epochs == 0:
            return save_path
        if save_path=='':
            return 'Specify a path where to store the Model'
        self._initialize(prediction_path)
            
        TMP_IOU=prediction_path+'TMP_IOU/'
        if not os.path.exists(TMP_IOU):
            os.makedirs(TMP_IOU)
     
        
        val_generator=Dataset_sat.from_root_folder(PATH_VALIDATION,self.nb_classes,max_data_size=500)
        val_loader = DataLoader(val_generator, batch_size=validation_batch_size,shuffle=False, num_workers=1)
        RBD=randint(0,int(val_loader.__len__())-1)
        
        
        train_generator=Dataset_sat.from_root_folder(PATH_TRAINING,self.nb_classes)
        train_loader = DataLoader(train_generator, batch_size=self.batch_size,shuffle=True, num_workers=1)
        if restore_path=='':
            loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(epochs,train_loader.__len__(),prediction_path,'w')
            print('Model trained from scratch')
        else:
            loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif=save_metrics(epochs,train_loader.__len__(),prediction_path,'a')
            self.net.load_state_dict(torch.load(restore_path))
            print('Model loaded from {}'.format(restore_path))
        
        self.store_init(val_loader,"_init",RBD)
        
        logging.info("Start optimization")

        counter=0
        
        for epoch in range(epochs):
            
            if epoch in reduce_lr_steps:
        
                self.lr = self.lr * 0.3
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

                ## Fwd+loss+bckwrd
                
                probs=predict(self.net,X,self.optimizer)
                self.optimizer.zero_grad()
                loss=criterion(Y,probs)
                loss.backward()
                self.optimizer.step()
                
                total_loss+=loss.data[0]
                loss_train[counter]=loss.data[0]
                file_train.write(str(loss_train[counter])+'\n')
                counter+=1
                
                if i_batch % display_step == 0:
                    self.output_training_stats(i_batch,X,Y)
                if counter % rec_save == 0:
                    torch.save(self.net.state_dict(),save_path + 'CP{}.pth'.format(counter))
                    print('Checkpoint {} saved !'.format(counter))

            logging.info(" Training {:}, Minibatch Loss= {:.4f}".format("epoch_%s"%epoch,total_loss/train_loader.__len__()))
            error_rate_v,loss_v,iou_v,iou_acc_v,f1_v=self.store_validation(val_loader,"epoch_%s"%epoch,RBD,file_gson=TMP_IOU,save_patches=False)
            IOU_verif[epoch]=iou_v
            IOU_acc_verif[epoch]=iou_acc_v
            f1_IOU_verif[epoch]=f1_v
            loss_verif[epoch]=loss_v
            
            IOU_file_verif.write(str(IOU_verif[epoch])+'\n')
            IOU_acc_file_verif.write(str(IOU_acc_verif[epoch])+'\n')
            f1_IOU_file_verif.write(str(f1_IOU_verif[epoch])+'\n')
            file_verif.write(str(loss_verif[epoch])+'\n')
            
        error_rate_v,loss_v,iou_v,iou_acc_v,f1_v=self.store_validation(val_loader,"epoch_%s"%epoch,RBD,file_gson=TMP_IOU,save_patches=True)
        
        return save_path, loss_train,loss_verif,IOU_verif,IOU_acc_verif,f1_IOU_verif      
    
    def output_training_stats(self, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        predictions=predict(self.net,batch_x,self.optimizer)
        loss=criterion(batch_y,predictions)
        loss=loss.data[0]
        predictions=predictions.data.cpu().numpy()
        groundtruth=batch_y.data.cpu().numpy()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Minibatch error= {:.1f}%".format(step,loss,error_rate(predictions, groundtruth)))
   
        
        
        
    def store_init(self,val_loader,name,random_batch_display,*,save_patches=True):
        loss_v=0
        error_rate_v=0

        for i_batch,sample in enumerate(val_loader):
            batch_x=standardize(sample['input'])
            batch_y=sample['groundtruth']
            
            X = Variable(batch_x.float())
            X=X.permute(0,3,1,2).cuda()  
            Y = Variable(batch_y.float())
            Y=Y.cuda()  
            
            probs=predict(self.net,X,self.optimizer)
            loss=criterion(Y,probs)
            
            loss_v+=loss.data[0]
            prediction_v=probs.data.cpu().numpy()
            groundtruth=np.asarray(batch_y)
            error_rate_v+=error_rate(prediction_v,groundtruth)
            if i_batch==random_batch_display and save_patches:
                batch_x=np.asarray(batch_x)
                if batch_x.shape[-1]==9:
                    pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)
                elif batch_x.shape[-1]==5 or batch_x.shape[-1]==17:
                    pansharp=np.stack((batch_x[:,:,:,3],batch_x[:,:,:,2],batch_x[:,:,:,1]),axis=3)
                plot_summary(prediction_v,groundtruth,pansharp,name,self.prediction_path,save_patches)

                         
        loss_v/=val_loader.__len__()   
        error_rate_v/=val_loader.__len__()  
        logging.info("Verification  loss= {:.4f},error= {:.1f}%".format(loss_v,error_rate_v))
        
    
    def store_validation(self,val_loader, name,random_batch_display,*,file_gson='',save_patches=True):
        loss_v=0
        iou_v=0
        iou_acc_v=0
        f1_v=0
        error_rate_v=0

        for i_batch,sample in enumerate(val_loader):
            batch_x=standardize(sample['input'])
            batch_y=sample['groundtruth']
            
            X = Variable(batch_x.float())
            X=X.permute(0,3,1,2).cuda()  
            Y = Variable(batch_y.float())
            Y=Y.cuda()  
            
            probs=predict(self.net,X,self.optimizer)
            loss=criterion(Y,probs)
            loss_v+=loss.data[0]
            
            prediction_v=probs.data.cpu().numpy()
            groundtruth=np.asarray(batch_y)
            iou_acc,f1,iou=predict_score_batch(file_gson,np.argmax(groundtruth,3),np.argmax(prediction_v,3))
            iou_acc_v+=iou_acc
            iou_v+=iou
            f1_v+=f1
            error_rate_v+=error_rate(prediction_v,groundtruth)
            if i_batch==random_batch_display and save_patches:
                batch_x=np.asarray(batch_x)
                if batch_x.shape[-1]==9:
                    pansharp=np.stack((batch_x[:,:,:,5],batch_x[:,:,:,3],batch_x[:,:,:,2]),axis=3)
                elif batch_x.shape[-1]==5 or batch_x.shape[-1]==17:
                    pansharp=np.stack((batch_x[:,:,:,3],batch_x[:,:,:,2],batch_x[:,:,:,1]),axis=3)
                plot_summary(prediction_v,groundtruth,pansharp,name,self.prediction_path,save_patches)

        loss_v/=val_loader.__len__()   
        iou_v/=val_loader.__len__()  
        iou_acc_v/=val_loader.__len__()  
        f1_v/=val_loader.__len__()  
        error_rate_v/=val_loader.__len__()  

        logging.info("Verification  loss= {:.4f},error rate= {:.1f}%, IOU = {:.4f}, IOU Precision = {:.4f}%, F1 IOU= {:.4f}%".format(loss_v,error_rate_v,iou_v,iou_acc_v,f1_v))


        return error_rate_v,loss_v,iou_v,iou_acc_v,f1_v
    
    
loss_fn=nn.CrossEntropyLoss()
def criterion(y,y_):
#     y=y.permute(0,3,1,2)
#     y_=y_.permute(0,3,1,2)
    
#     loss=Fu.binary_cross_entropy_with_logits(y_,y)
    
    y = y.contiguous().view(-1,y.size()[-1])
    y_ = y_.contiguous().view(-1,y.size()[-1])
    y = y.max(-1)[1]
    loss = loss_fn( y_,y)
    
    return loss


def predict(net,batch_x,optimizer):
    
#     optimizer.zero_grad()
#     logits=net.forward(batch_x) 
    logits=net(batch_x) 
    probs=logits.permute(0,2,3,1)
    return probs


def save_metrics(epochs,training_len,prediction_path,mode):
    #STORE loss for ANALYSIS
    loss_train=np.zeros(epochs*training_len)
    file_train = open(prediction_path+'loss_train.txt',mode) 
    loss_verif=np.zeros(epochs)
    file_verif = open(prediction_path+'loss_verif.txt',mode) 
    #STORE IOU for ANALYSIS
    IOU_verif=np.zeros(epochs)
    IOU_file_verif = open(prediction_path+'iou_verif.txt',mode)
    #STORE IOU_ACC for ANALYSIS
    IOU_acc_verif=np.zeros(epochs)
    IOU_acc_file_verif = open(prediction_path+'iou_acc_verif.txt',mode)
    #STORE f1_IOU for ANALYSIS
    f1_IOU_verif=np.zeros(epochs)
    f1_IOU_file_verif = open(prediction_path+'f1_iou_verif.txt',mode) 
    
    return loss_train,file_train,loss_verif,file_verif,IOU_verif,IOU_file_verif,IOU_acc_verif,IOU_acc_file_verif,f1_IOU_verif,f1_IOU_file_verif
def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))
def plot_summary(predictions,labels,pansharp,epoch,prediction_path,save_patches):
    
#     fig,axs=plt.subplots(3, len(pansharp),figsize=(8*len(pansharp),24))

#     axs[0,0].set_title(epoch+' Pansharpened ', fontsize='large')
#     axs[1,0].set_title(epoch+' Groundtruth ', fontsize='large')
#     axs[2,0].set_title(epoch+' Predictions ', fontsize='large')

    labels=np.argmax(labels, 3) 
    logits=np.argmax(predictions, 3)

    for i in range(len(pansharp)):

#         axs[0,i].imshow(pansharp[i])
#         axs[1,i].imshow(labels[i]) 
#         axs[2,i].imshow(logits[i])
        
        
        if save_patches:
            plt.imsave(prediction_path+epoch+'_Panchro_'+str(i)+'.jpg',pansharp[i])
            plt.imsave(prediction_path+epoch+'_Groundtruth_'+str(i)+'.jpg',labels[i])
            plt.imsave(prediction_path+epoch+'_Predictions_'+str(i)+'.jpg',logits[i])

#     plt.subplots_adjust()
#     plt.show()
    
    
    
if __name__ == '__main__':
    
    
      #python resunet_main.py ../2_DATA_GHANA/DATASET/120_x_120_8_pansh/ MODEL_BASIC_TEST_120/ RESUNET_val_spacenet.ckpt '' --input_channels=9 --nb_classes=2 --nb_layers=3 --nb_features_root=32  --learning_rate=0.0001 --batch_size=8  --epochs=1 --dropout=0.1 --display_step=100 --validation_size_batch=100 --rec_save_model=400

# python resunet_main.py /scratch/SPACENET_DATA_PROCESSED/DATASET/120_x_120_8_bands_pansh/ MODEL_VAL_SPACENET/ RESUNET_pytorch_val_spacenet_restored.ckpt MODEL_VAL_SPACENET/RESUNET_pytorch_val_spacenet.ckptCP190000.pth --input_channels=9 --nb_classes=2 --nb_layers=3 --nb_features_root=32  --learning_rate=1e-4 --batch_size=32  --epochs=100 --dropout=0.1 --display_step=100 --validation_size_batch=100 --rec_save_model=2000

#python resunet_main.py ../2_DATA_GHANA/DATASET/120_x_120_8_pansh/ MODEL_MELI_GHANA/ RESUNET_meli_ghana_restored2.ckpt MODEL_MELI_GHANA/RESUNET_pytorch_meli_ghana_restored.ckptCP30000.pth --input_channels=9 --nb_classes=2 --nb_layers=3 --nb_features_root=32  --learning_rate=1e-5 --batch_size=8  --epochs=100 --dropout=0.1 --display_step=100 --validation_size_batch=100 --rec_save_model=2000
    
    root_folder=sys.argv[1]
#     root_folder = '../2_DATA_GHANA/DATASET/120_x_120_8_pansh/'
#     root_folder ='/scratch/SPACENET_DATA_PROCESSED/DATASET/120_x_120_8_bands_pansh/'

    
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
        else:
            raise ValueError('Unknown argument %s' % str(arg))
    
    model=UNet(INPUT_CHANNELS,NB_CLASSES,DEFAULT_LAYERS,DEFAULT_FEATURES_ROOT,DROPOUT)
    model.apply(weights_init)
    model.cuda()
    cudnn.benchmark = True
    trainer=Trainer(model,DEFAULT_BATCH_SIZE,DEFAULT_LR,NB_CLASSES)
    save_path,loss_train,loss_verif,iou_verif,iou_acc_verif,f1_iou_verif=trainer.train( root_folder, MODEL_PATH_SAVE, MODEL_PATH_RESTORE,DEFAULT_EPOCHS,DROPOUT, DISPLAY_STEP, DEFAULT_VALID,REC_SAVE, TEST_SAVE)
    print('Last model saved is %s: '%save_path)
#     #SAVE loss
#     plt.title('Plot Loss', fontsize=20)
#     ite = np.arange(0,len(loss_train),1)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     plt.plot(ite,loss_train,'b',epo,loss_verif,'g')
#     plt.ylabel('Loss')
#     plt.show()
    
#      #SAVE IOU
#     plt.title('Plot IOU', fontsize=20)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     plt.plot(epo,iou_verif,'g')
#     plt.ylabel('IOU in %')
#     plt.show()
    
#     #SAVE IOU  acc
#     plt.title('Plot IOU Accuracy', fontsize=20)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     plt.plot(epo,iou_acc_verif,'g')
#     plt.ylabel('IOU Accuracy in %')
#     plt.show()


    
#      #SAVE f1 IOU
#     plt.title('Plot f1 IOU', fontsize=20)
#     epo=np.arange(int(len(loss_train)/DEFAULT_EPOCHS)-1,len(loss_train),int(len(loss_train)/DEFAULT_EPOCHS))
#     plt.plot(epo,f1_iou_verif,'g')
#     plt.ylabel('f1 IOU in %')
#     plt.show()
