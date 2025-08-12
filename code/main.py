from src.Unet import Unet_512
from src.Fresnal import Fresnal
from src.utils import preprocess_image,show_image_tensor,show_MSE_loss_distribution,calculate_ssim,calculate_psnr
from src.loss_func import com_loss
from src.train_test import test
from src.Fresnal_extra import Fresnal_ex

from Ui.Ui_design import Ui_MainWindow
from Ui.Ui_input import Ui_Input
from Ui.Ui_aper import Ui_Aperture
from Ui.Ui_rec import Ui_Reconstruct
from Ui.Ui_stream import Ui_Stream
from Ui.Ui_Input_Dialog import Ui_Dialog
from Ui.Ui_train_dialog import Ui_Training_Diag
from Ui.Ui_set_train import Ui_train_settiings
from Ui.Ui_More import Ui_more
from Ui.Ui_Loss_window import Ui_Loss_window
from Ui.Ui_MSE_Window import Ui_mse_window
from Ui.Ui_histWindow import Ui_HistWindow
from Ui.Ui_ininverse import Ui_ForwDif
from Ui.Ui_resizelog import Ui_Resizelog

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt,Signal,QThread)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QSizePolicy, QVBoxLayout,
    QWidget,QMainWindow,QFileDialog,QMessageBox,QDialog)

import numpy as np
from PIL import Image,ImageQt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tensor_to_qpixmap_method2(tensor, target_size=(351, 351)):
        """
        方法2：使用PIL进行尺寸调整
        
        Args:
            tensor: 四维张量 (1, 1, 512, 512)
            target_size: 目标尺寸 (width, height)
        
        Returns:
            QPixmap对象
        """
        # 1. 移除批次和通道维度
        if tensor.dim() == 4:
            image_data = tensor.squeeze(0).squeeze(0)
        
        # 2. 转换为numpy数组并归一化
        if isinstance(tensor, torch.Tensor):
            image_array = image_data.detach().cpu().numpy()
        else:
            image_array = image_data
        
        # 3. 数据归一化
        if image_array.dtype != np.uint8:
            # if image_array.min() >= 0 and image_array.max() <= 1:
            #     image_array = (image_array * 255).astype(np.uint8)
            # elif image_array.min() >= -1 and image_array.max() <= 1:
            #     image_array = ((image_array + 1) / 2 * 255).astype(np.uint8)
            # else:
                image_array = ((image_array - image_array.min()) / 
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # 4. 使用PIL调整尺寸
        pil_image = Image.fromarray(image_array, mode='L')  # 'L' 表示灰度图像
        resized_pil = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 5. 转换回numpy数组
        resized_array = np.array(resized_pil)
        
        # 6. 创建QImage和QPixmap
        height, width = resized_array.shape
        qimage = QImage(resized_array.data, width, height, width, QImage.Format_Grayscale8)
        qpixmap = QPixmap.fromImage(qimage)
        
        return qpixmap

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1140,850)
        #model data
        self.mode=1
        self.rate=8e-6
        self.lamb=632e-9
        self.d=0.4
        self.model=None
        self.optimizer=None
        self.kl1=0.1
        self.kMSE=0.4
        self.kfreq=0.5
        self.kssim=0.5
        self.kpsnr=0.5
        self.criterion=com_loss(self.kl1,self.kMSE,self.kfreq,self.kpsnr,self.kssim)
        self.num_epochs=1500
        self.patience=500
        self.image=None
        self.losses=None
        self.best_k=None
        self.ssim_loss=None
        self.psnr_loss=None
        self.model=Unet_512(mode=self.mode,d=self.d,rate=self.rate,lambda_=self.lamb).to(DEVICE)
        self.img_size=512

        self.hist_item=None
        self.result_save_path="results"
        self.buffer_save_path='buffer'
        self.others_name="details"
        self.result_path=None
        self.buffer_path=None
        self.aperture_name="aperture.png"
        self.reconstruct_orig_name="reconstructed_original.png"
        self.reconstruct_light_name="reconstructed_stronger_light.png"
        self.init_name="init.txt"
        self.origin_name='origin.png'
        self.loss_name='loss.png'
        self.mse_name='mse_visualize.png'
        #ui data
        Path(self.result_save_path).mkdir(exist_ok=True)
        Path(self.buffer_save_path).mkdir(exist_ok=True)
        self.setupUi(self)
        #self.check_hist()
        self.create_pages()
        self.bind()
    
    # def check_hist(self):
    #     for name in os.listdir(self.result_save_path):
    #         self.his_list.addItem(str(name))
            
    def create_pages(self):
        #input page
        self.input_page=InputWindow()
        self.input_page.dlb.setText(str(round(self.d,3)))
        self.input_page.lamblb.setText(str(round(self.rate*1000000,3)))
        self.input_page.ratelb.setText(str(round(self.lamb*1000000000,1)))
        self.input_page.pic_lb.setPixmap(QPixmap("Ui/pic/pic2.png").scaled(471,331))
        self.input_page.modelb.setText("fresnel,amplitude")
        self.stackedWidget.addWidget(self.input_page)
        #aperture page
        self.aper_page=AperWindow()
        self.stackedWidget.addWidget(self.aper_page)
        #recon page
        self.rec_page=RecWindow()
        self.stackedWidget.addWidget(self.rec_page)

    def switch_widget(self,index):
        self.stackedWidget.setCurrentIndex(index)


    def bind(self):
        #main window
        self.input_btn.clicked.connect(lambda:self.switch_widget(0))
        self.aper_btn.clicked.connect(lambda:self.switch_widget(1))
        self.rec_btn.clicked.connect(lambda:self.switch_widget(2))
        self.more_btn.clicked.connect(self.get_more)
        
        self.actionload.triggered.connect(self.load_image)
        self.actioninitial.triggered.connect(self.upload)
        self.actiontraining.triggered.connect(self.set_train)
        self.actionhist.triggered.connect(self.read_history)
        self.actionFresne_amplitude.triggered.connect(lambda:self.change_mode(1))
        self.actionFresnel_phase.triggered.connect(lambda:self.change_mode(2))
        self.actionFraunhofer_amplitude.triggered.connect(lambda:self.change_mode(3))

        self.actionForward_diffraction_stimulator.triggered.connect(self.forward_dif)
        # self.his_list.itemDoubleClicked.connect(self.read_history)
        
        
        #input page
        self.input_page.start_btn.clicked.connect(self.generate)
        
        #self.aper_page.reset_btn.connect(self.generate)
        
        #rec_page
        self.rec_page.orig_radioButton.setChecked(True)
        self.rec_page.orig_radioButton.clicked.connect(lambda:self.display_image("rec_page",os.path.join(self.result_path,self.reconstruct_orig_name)))
        self.rec_page.light_radioButton.clicked.connect(lambda:self.display_image("rec_page",os.path.join(self.result_path,self.reconstruct_light_name)))
    
    def forward_dif(self):
        win=DifWindow()
        win.show()


    def change_mode(self,index):
        self.mode=index
        if index==1:
            self.input_page.modelb.setText("fresnel,amplitude")
        if index==2:
            self.input_page.modelb.setText("fresnel,phase")
        if index==3:
            self.input_page.modelb.setText("fraunhofer,amplitude")
    
    def get_more(self,state=0):
        
        if state==0:
            base_dir=self.result_path
        if state==1:
            base_dir=self.buffer_path
        dir=os.path.join(base_dir,self.others_name)
        mse_path=os.path.join(dir,self.mse_name)
        loss_path=os.path.join(dir,self.loss_name)
        self.more_dialog=More(mse_path=mse_path,loss_path=loss_path,ssim_loss=self.ssim_loss,psnr_loss=self.psnr_loss)
        try:
            origin_path=os.path.join(base_dir,self.origin_name)
            aper_path=os.path.join(base_dir,self.aperture_name)
            rec_path=os.path.join(base_dir,self.reconstruct_light_name)
            rec_origin_path=os.path.join(base_dir,self.reconstruct_orig_name)

            ori=QPixmap(origin_path).scaled(
                #self.more_dialog.dis_input_lb.size(),
                QSize(331,331),
                Qt.KeepAspectRatio,  # 保持长宽比
                Qt.SmoothTransformation  # 平滑变换
            )
            aper=QPixmap(aper_path).scaled(
                #self.more_dialog.dis_aper_lb.size(),
                QSize(331,331),
                Qt.KeepAspectRatio,  # 保持长宽比
                Qt.SmoothTransformation  # 平滑变换
            )
            rec=QPixmap(rec_path).scaled(
                #self.more_dialog.dis_rec_lb.size(),
                QSize(331,331),
                Qt.KeepAspectRatio,  # 保持长宽比
                Qt.SmoothTransformation  # 平滑变换
            )
            rec_origin=QPixmap(rec_origin_path).scaled(
                QSize(331,331),
                #self.more_dialog.dis_origin_lb.size(),
                Qt.KeepAspectRatio,  # 保持长宽比
                Qt.SmoothTransformation  # 平滑变换
            )
            # ori=QPixmap(origin_path)
            # rec_origin=QPixmap(rec_origin_path)
            # rec=QPixmap(rec_path)
            # aper=QPixmap(aper_path)

            self.more_dialog.dis_origin_lb.setPixmap(rec_origin)
            self.more_dialog.dis_rec_lb.setPixmap(rec)
            self.more_dialog.dis_input_lb.setPixmap(ori)
            self.more_dialog.dis_aper_lb.setPixmap(aper)
        except:
            QMessageBox.critical(self,"Error","No Existing Results")
        self.more_dialog.show()

    def set_hist_item(self,item):
        self.hist_item=item
        try:
            self.log_edit.append(f"选择历史记录：{self.hist_item}")
            self.result_path=os.path.join(self.result_save_path,self.hist_item)
            others_path=os.path.join(self.result_path,self.others_name)
            self.image=preprocess_image(os.path.join(self.result_path,self.origin_name),self.img_size).to(DEVICE)
            
            

            with open(os.path.join(others_path,self.init_name), 'r') as f:
                lines = f.readlines()
                self.d = float(lines[0].strip())
                self.lamb = float(lines[1].strip())
                self.rate = float(lines[2].strip())
                self.kl1=float(lines[3].strip())
                self.kMSE = float(lines[4].strip())
                self.kfreq = float(lines[5].strip())
                self.kssim = float(lines[6].strip())
                self.kpsnr = float(lines[7].strip())
                self.ssim_loss=float(lines[8].strip())
                self.psnr_loss=float(lines[9].strip())
                self.mode=float(lines[10].strip())
                self.input_page.dlb.setText(str(self.d))
                self.input_page.lamblb.setText(str(self.lamb*1000000000))
                self.input_page.ratelb.setText(str(self.rate*1000000))
                self.change_mode(self.mode)

            
        except:
            QMessageBox.critical(self,"Error","Invalid History")
            return None
        self.display_image("aper_page",os.path.join(self.result_path,self.aperture_name))
        self.display_image("input_page",os.path.join(self.result_path,self.origin_name))
        self.display_image("rec_page",os.path.join(self.result_path,self.reconstruct_orig_name))
    def read_history(self):
        self.histwind=HistWindow(self.result_save_path)
        self.histwind.show()
        self.histwind.read.connect(self.set_hist_item)
        


    def load_image(self):
        img_path=QFileDialog.getOpenFileName(self,"choose an image","./","(*.jpg *.png *.jpeg)")[0]
        try:
            img_show=Image.open(img_path)
            img_show=img_show.resize((self.img_size,self.img_size))
            self.input_page.display_lb.setPixmap(ImageQt.toqpixmap(img_show))
            self.image=preprocess_image(img_path,self.img_size).to(DEVICE)
        except:
            QMessageBox.critical(self,'Error',"Invalid Image Input")
            return None
        self.model=None
        self.stackedWidget.setCurrentIndex(0)
        self.aper_page.display_lb.clear()
        self.rec_page.display_lb.clear()
        self.log_edit.append(f"loaded file:{img_path}")
        #print(self.image)
    def set_train(self):
        dialog=SetTrain()
        dialog.num_epoch_edit.setText(str(self.num_epochs))
        dialog.pat_edit.setText(str(self.patience))
        dialog.mse_edit.setText(str(self.kMSE))
        dialog.freq_edit.setText(str(self.kfreq))
        dialog.l1_edit.setText(str(self.kl1))
        dialog.SSMI_edit.setText(str(self.kssim))
        dialog.PSNR_edit.setText(str(self.kpsnr))
        if dialog.exec()==QDialog.Accepted:
            #print("Acept")
            try:
                a=int(dialog.num_epoch_edit.text())
                b=int(dialog.pat_edit.text())
                c=float(dialog.mse_edit.text())
                d=float(dialog.freq_edit.text())
                e=float(dialog.l1_edit.text())
                f=float(dialog.SSMI_edit.text())
                g=float(dialog.PSNR_edit.text())
            except:
                QMessageBox.critical(self,'Error',"Invalid Input!")
                self.log_edit.append("Invalid input!")
                return None
            if a<1 or b<1:
                QMessageBox.critical(self,'Error',"Invalid Input!")
                return None
            if c<0 or d<0 or e<0:
                QMessageBox.critical(self,'Error',"Invalid Input!")
                return None
            self.num_epochs=a
            self.patience=b
            self.kMSE=c
            self.kfreq=d
            self.kl1=e
            self.kssim=f
            self.kpsnr=g

            self.input_page.epoch_lb.setText(str(self.num_epochs))
            
    def upload(self):
        dialog=Input_Dialog()
        dialog.dedit.setText(str(round(self.d,3)))
        dialog.rateEdit.setText(str(round(self.rate*1000000,3)))
        dialog.lambEdit.setText(str(round(self.lamb*1000000000,1)))
        #dialog.dedit.textChanged.connect(dialog.update_images)
        if dialog.exec()==QDialog.Accepted:
            # print("Acept")
            # dialog.get_data()
            try:
                d=float(dialog.dedit.text())
                rate=float(dialog.rateEdit.text())
                lamb=float(dialog.lambEdit.text())
                #dialog.update_images(d*10)
                print(d,rate,lamb)
            except:
                a=QMessageBox.critical(self,'Error',"Invalid input!")
                self.log_edit.append("Invalid input!")
                return None
            if d<=0.001:
                QMessageBox.critical(self,'Error',"Invalid d(too small)!")
                return None
            self.input_page.dlb.setText(str(d))
            self.input_page.lamblb.setText(str(lamb))
            self.input_page.ratelb.setText(str(rate))
            self.d=d
            self.lamb=lamb*0.000000001
            self.rate=rate*0.000001
        
    
    def set_model(self,best_model):
        self.model.load_state_dict(best_model)
        print('set')
        
    def set_losses(self,losses):
        self.losses=losses
        
    def set_best_k(self,best_k):
        self.best_k=best_k
        
    def append_progress(self,text):
        self.log_edit.append(text)
        self.dialog.textEdit.append(text)
    
    def save_buffer(self,name):
        buffer_name=name
        self.save_results(model_=self.model,img=self.image,state=1,buff_name=buffer_name)
        self.get_more(state=1)

    def generate(self):
        if self.image==None:
            QMessageBox.critical(self,"Error","No Input Image")
            return None
        self.dialog=Train_Dialog()
        Path(self.buffer_save_path).mkdir(exist_ok=True)
        self.criterion=com_loss(self.kl1,self.kMSE,self.kfreq,self.kpsnr,self.kssim)
        self.dialog.show()
        self.train_thread=TrainThread(mode=self.mode,
                                      image=self.image
                                        ,d=self.d
                                        ,lamb=self.lamb
                                        ,rate=self.rate
                                        ,epoch=self.num_epochs
                                        ,criterion=self.criterion
                                        ,patience=self.patience)
        self.train_thread.stop_flag=False
        self.model=Unet_512(mode=self.mode,d=self.d,rate=self.rate,lambda_=self.lamb).to(DEVICE)
        self.criterion=com_loss(self.kl1,self.kMSE,self.kfreq,self.kpsnr,self.kssim)
        #print("线程对象：", self.train_thread)
        self.dialog.stop_btn.clicked.connect(self.train_thread.stop)
        self.train_thread.progress.connect(self.append_progress)
        self.train_thread.result_model.connect(self.set_model)
        self.train_thread.result_loss.connect(self.set_losses)
        self.train_thread.result_k.connect(self.set_best_k)
        self.train_thread.finished.connect(self.finish_training)
        self.train_thread.interrupt.connect(self.log_edit.append)
        self.train_thread.save_buff.connect(self.save_buffer)
        
        self.input_page.start_btn.setEnabled(False)
        self.input_page.start_btn.setText("训练中")
        self.log_edit.append("开始训练...")
        self.log_edit.append(f"使用设备: {DEVICE}")
        self.log_edit.append(f"图像形状: {self.image.squeeze(0).squeeze(0).shape}")
        self.log_edit.append(f"使用条件：d={self.d},波长={self.lamb},rate={self.rate}")
        self.train_thread.start()
        

    def display_image(self,position,image_path):
        try:
            img=Image.open(image_path)
        except:
            QMessageBox.critical(self,"Error","Can't find corresponding image")
        if position=="aper_page":
            self.aper_page.display_lb.setPixmap(ImageQt.toqpixmap(img))
        if position=="rec_page":
            self.rec_page.display_lb.setPixmap(ImageQt.toqpixmap(img))
        if position=='input_page':
            self.input_page.display_lb.setPixmap(ImageQt.toqpixmap(img))

    
    
    def finish_training(self):
        self.log_edit.append("训练完毕")
        self.log_edit.append(f"best_k:{self.best_k}")
        self.dialog.close()
        self.dialog.deleteLater()
        self.input_page.start_btn.setEnabled(True)
        self.input_page.start_btn.setText("开始训练")
        self.save_results(model_=self.model,img=self.image)
        self.get_more()
        shutil.rmtree(self.buffer_save_path)

    def save_results(self,model_,img,state=0,buff_name=None):
        if state==0:
            dir_path=self.result_save_path
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir=os.path.join(dir_path,f"results_{current_time}")
            others_dir=os.path.join(base_dir,self.others_name)
            self.result_path=base_dir
            os.mkdir(base_dir)
            os.mkdir(others_dir)
        
        if state==1:
            dir_path=self.buffer_save_path
            base_dir=os.path.join(dir_path,buff_name)
            others_dir=os.path.join(base_dir,self.others_name)
            self.buffer_path=base_dir
            os.mkdir(base_dir)
            os.mkdir(others_dir)

        image=model_(img)[1].detach()
        k=model_(img)[2].detach()
        aper=model_(img)[0].detach()
        if self.mode==1:
            fres=Fresnal(512,lambda_=self.lamb,d=self.d,rate=self.rate)
        if self.mode==2:
            fres=Fresnal_ex(512,lambda_=self.lamb,d=self.d,rate=self.rate)
        if self.mode==3:
            print("not done yet")
            return None
        rec_origin=fres(aper)
        torchvision.utils.save_image(img,os.path.join(base_dir,self.origin_name),normalize=False)
        torchvision.utils.save_image(rec_origin,os.path.join(base_dir,self.reconstruct_orig_name),normalize=False)
        torchvision.utils.save_image(image,os.path.join(base_dir,self.reconstruct_light_name),normalize=True)
        torchvision.utils.save_image(aper,os.path.join(base_dir,self.aperture_name),normalize=False)
        self.display_image("aper_page",os.path.join(base_dir,self.aperture_name))
        self.display_image("rec_page",os.path.join(base_dir,self.reconstruct_orig_name))
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(others_dir,self.loss_name))
        plt.close()
        
        self.ssim_loss=calculate_ssim(img,image)
        self.psnr_loss=calculate_psnr(img,image)
        
        with open(os.path.join(others_dir,self.init_name), 'w') as f:
            f.write(f"{self.d}\n")
            f.write(f"{self.lamb}\n")
            f.write(f"{self.rate}\n")
            f.write(f"{self.kl1}\n")
            f.write(f"{self.kMSE}\n")
            f.write(f"{self.kfreq}\n")
            f.write(f"{self.kssim}\n")
            f.write(f"{self.kpsnr}\n")
            f.write(f"{self.ssim_loss}\n")
            f.write(f"{self.psnr_loss}\n")
            f.write(f"{self.mode}\n")
        show_MSE_loss_distribution(img.squeeze(0).squeeze(0),image.squeeze(0).squeeze(0),os.path.join(others_dir,self.mse_name))
        

        self.log_edit.append(f"Results saved to:{base_dir}")
        #self.his_list.addItem(f"results_{current_time}")

        
        return base_dir

class TrainThread(QThread):
    progress = Signal(str)  # 自定义信号，用于向主线程发送进度信息
    finished=Signal(str)
    result_model=Signal(object)
    result_k=Signal(object)
    result_loss=Signal(object)
    interrupt=Signal(object)
    save_buff=Signal(object)
    def __init__(self,image,mode,d,rate,lamb,epoch,criterion,patience):
        super().__init__()
        self.d=d
        self.rate=rate
        self.lamb=lamb
        self.num_epoch=epoch
        self.criterion=criterion
        self.patience=patience
        self.image=image
        self.mode=mode
        # 模拟训练过程
        self.stop_flag=False

    def stop(self):
        self.stop_flag=True

    def run(self):
        self.model=Unet_512(self.mode,d=self.d,rate=self.rate,lambda_=self.lamb).to(DEVICE)
        self.optimizer=optim.Adam(self.model.parameters(),lr=0.001)
        self.model.train()
        self.buffer=0
        best_model=None
        best_k=None
        count=0
        losses = []
        min_loss=100
        print(f'usring:{self.mode}')
        for epoch in range(self.num_epoch):
            if epoch%30==2:
                self.buffer+=1
                self.result_model.emit(self.model.state_dict())
                self.result_loss.emit(losses)
                self.save_buff.emit(str(self.buffer))
            
            self.optimizer.zero_grad()  # 每个epoch开始时清零梯度
            
            predicted_aperture, reconstructed_diffraction,k = self.model(self.image)
            loss = self.criterion(reconstructed_diffraction, self.image)
            
            loss.backward()
            self.optimizer.step()

            if loss.item()<min_loss:
                min_loss=loss.item()
                best_model=self.model.state_dict()
                best_k=k
                count=0
            else:
                count=count+1
            
            losses.append(loss.item())
            
        
            if (epoch + 1) % 10 == 0:
                self.progress.emit(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {loss.item():.6f}')
            if count>self.patience:
                break
            if self.stop_flag==True:
                self.interrupt.emit("用户终止")
                print("stopped")
                break
        self.result_loss.emit(losses)
        self.result_k.emit(best_k)
        self.result_model.emit(best_model)
        self.finished.emit("训练完毕") 
    
class InputWindow(QWidget,Ui_Input):
    def __init__(self):
        super().__init__()
        self.setupUi(self)     

class AperWindow(QWidget,Ui_Aperture):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 

class RecWindow(QWidget,Ui_Reconstruct):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 

class SetTrain(QDialog,Ui_train_settiings):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class Train_Dialog(QDialog,Ui_Training_Diag):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("模型训练中")
        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |  # 保留标题
            Qt.WindowStaysOnTopHint  # 置顶
        )
        self.setModal(True)  # 强制模态
        self.setWindowModality(Qt.ApplicationModal)  # 应用级模态
    def closeEvent(self, event):
        # 拦截手动关闭操作
        event.ignore()


class Input_Dialog(QDialog,Ui_Dialog):
    def __init__(self,parent=None):
        super().__init__()
        self.setupUi(self)
        self.horizontalSlider.setRange(10,4000)
        self.horizontalSlider.setValue(1580)
        self.horizontalSlider.valueChanged.connect(self.update_d)
        self.horizontalSlider_3.setRange(1,6000)
        self.horizontalSlider_3.setValue(6000)
        self.horizontalSlider_3.valueChanged.connect(self.update_rate)
        
        self.dedit.editingFinished.connect(self.update_slider1)
        self.lambEdit.editingFinished.connect(self.update_slider2)
        self.rateEdit.editingFinished.connect(self.update_slider3)

        #self.verticalSlider.valueChanged.connect(self.update_lamb)
        self.label.setPixmap(QPixmap(u"Ui/pic/pic2.png").scaled(481,351))
        # self.label.setPixmap(QPixmap(u"Ui/pic/aperture.png"))
        # self.label_2.setPixmap(QPixmap(u"Ui/pic/light.png"))
        # self.label_3.setPixmap(QPixmap(u"Ui/pic/result.png"))
        # self.original_pixmap_2=QPixmap(u"Ui/pic/light.png")
        # self.initial_width_1=103
        # self.initial_width_2=85
        # self.initial_width_3=78
        # self.initial_height=130
        # self.initial_pos_1=(42,45)
        # self.initial_pos_2=(145,45)
        # self.initial_pos_3=(230,45)

    def update_d(self):
        value=self.horizontalSlider.value()*0.001
        value=round(value,3)
        self.dedit.setText(str(value))

    def update_lamb(self):
        value=self.verticalSlider.get_wavelength()
        # value=round(value,3)
        self.lambEdit.setText(str(value))

    def update_rate(self):
        value=self.horizontalSlider_3.value()*0.01
        value=round(value,2)
        self.rateEdit.setText(str(value))
    
    def update_slider1(self):
        try:
            value=float(self.dedit.text())*1000
            self.horizontalSlider.setValue(value)
        except:
            return None
        
    def update_slider2(self):
        try:
            value=float(self.lambEdit.text())*10
            self.verticalSlider.setValue(value)
        except:
            return None
        
    def update_slider3(self):
        try:
            value=float(self.rateEdit.text())*100
            self.horizontalSlider_3.setValue(value)
        except:
            return None

    # def update_images(self, value):
    #     """根据滑块值更新图片位置和大小"""
    #     # # 图片1保持不动
    #     # self.label1.setGeometry(
    #     #     self.initial_pos1[0], 
    #     #     self.initial_pos1[1], 
    #     #     self.initial_size1[0], 
    #     #     self.initial_size1[1]
    #     # )
        
    #     # 图片2水平拉伸
    #     try:
    #         value=float(value)
    #         value=value-1.58
    #         #value=int(value)
    #         value=value*30
    #         if value>80:
    #             value=80
    #     except:
    #         return None
    #     new_width2 = self.initial_width_2 + value
    #     print(self.initial_pos_2[0]+new_width2)
    #     self.label_2.setGeometry(
    #          self.initial_pos_2[0], 
    #          self.initial_pos_2[1], 
    #          new_width2, 
    #          self.initial_height
    #      )
        
    #     # 重新缩放图片2以适应新宽度
    #     scaled_pixmap2 = self.original_pixmap_2.scaled(
    #         new_width2, 
    #         self.initial_height, 
    #         Qt.IgnoreAspectRatio, 
    #         Qt.SmoothTransformation
    #     )
    #     self.label_2.setPixmap(scaled_pixmap2)
        
    #     # 图片3水平向右移动
    #     new_x3 = self.initial_pos_3[0] + value
    #     print(new_x3)
    #     self.label_3.setGeometry(
    #         new_x3, 
    #         self.initial_pos_3[1], 
    #         self.initial_width_3,
    #         self.initial_height
    #         )

    # def get_data(self):
    #     try:
    #         d=float(self.dedit.text())
    #         rate=float(self.rateEdit.text())
    #         lamb=float(self.lambEdit.text())
    #         print(d,rate,lamb)
    #         return d,rate,lamb
    #     except:
    #         a=QMessageBox.critical(self,'Error',"Invalid input!")
    #         return None

class More(QWidget,Ui_more):
    def __init__(self,mse_path,loss_path,ssim_loss,psnr_loss):
        super().__init__()
        self.setupUi(self)
        self.mse_btn.clicked.connect(self.call_mse)
        self.loss_btn.clicked.connect(self.call_loss)
        self.mse_path=mse_path
        self.loss_path=loss_path
        self.ssim_loss=ssim_loss
        self.psnr_loss=psnr_loss

        self.ssimlb.setText(f"SSIM:{self.ssim_loss}")
        self.psnrlb.setText(f"PSNR:{self.psnr_loss}")

        # self.dis_input_lb.setSizePolicy(
        #     QSizePolicy.Expanding, 
        #     QSizePolicy.Expanding
        # )
        # self.dis_rec_lb.setSizePolicy(
        #     QSizePolicy.Expanding, 
        #     QSizePolicy.Expanding
        # )
        # self.dis_origin_lb.setSizePolicy(
        #     QSizePolicy.Expanding, 
        #     QSizePolicy.Expanding
        # )
        # self.dis_aper_lb.setSizePolicy(
        #     QSizePolicy.Expanding, 
        #     QSizePolicy.Expanding
        # )
    def call_mse(self):
        self.mse=MseWindow()
        img=Image.open(self.mse_path)
        self.mse.dis_lb.setPixmap(ImageQt.toqpixmap(img))
        self.mse.show()

    def call_loss(self):
        self.loss_=LossWindow()
        img=Image.open(self.loss_path)
        self.loss_.dis_lb.setPixmap(ImageQt.toqpixmap(img))
        self.loss_.ssmi_lb.setText(str(self.ssim_loss))
        self.loss_.psnr_lb.setText(str(self.psnr_loss))
        self.loss_.show()



class MseWindow(QDialog,Ui_mse_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class LossWindow(QDialog,Ui_Loss_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class ResizeWind(QDialog,Ui_Resizelog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
class HistWindow(QDialog,Ui_HistWindow):
    read=Signal(str)
    def __init__(self,hist_path):
        super().__init__()
        self.setupUi(self)
        for name in os.listdir(hist_path):
            self.hist_list.addItem(str(name))
        
        self.hist_list.itemDoubleClicked.connect(self.reading)

    def reading(self,item):
        self.read.emit(item.text())

class DifWindow(QWidget,Ui_ForwDif):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.d=0.6
        self.rate=60e-6
        self.lambda_=632e-9
        self.mode=0
        self.img_size=512
        self.img_size_H=512
        self.image=None
        self.rec=None
        self.model=Fresnal(self.img_size,lambda_=self.lambda_,d=self.d,rate=self.rate).to(DEVICE)
        self.result_path="forward_diffraction"
        self.current_result_path=None
        self.aper_name='aperture.png'
        self.rec_name='pattern.png'
        self.init_name='init.txt'
        

        self.horizontalSlider.setRange(10,4000)
        self.horizontalSlider.setValue(1580)
        self.horizontalSlider.valueChanged.connect(self.update_d)
        self.horizontalSlider_3.setRange(1,6000)
        self.horizontalSlider_3.setValue(6000)
        self.horizontalSlider_3.valueChanged.connect(self.update_rate)
        
        self.dedit.editingFinished.connect(self.update_slider1)
        self.lambEdit.editingFinished.connect(self.update_slider2)
        self.rateEdit.editingFinished.connect(self.update_slider3)

        self.resizebtn.clicked.connect(self.update_size)
        self.startbtn.clicked.connect(self.generate)
        self.inputbtn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_results)
        self.comboBox.currentIndexChanged.connect(lambda index:self.changeMode(index=index))

        #self.verticalSlider.valueChanged.connect(self.update_lamb)

        self.dedit.setText(str(round(self.d,3)))
        self.rateEdit.setText(str(round(self.rate*1000000,3)))
        self.lambEdit.setText(str(round(self.lambda_*1000000000,1)))
        Path(self.result_path).mkdir(exist_ok=True)
        self.label_4.setPixmap(QPixmap("Ui/pic/pic2.png").scaled(391,291,Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    def update_d(self):
        value=self.horizontalSlider.value()*0.001
        value=round(value,3)
        self.dedit.setText(str(value))

    def update_lamb(self):
        value=self.verticalSlider.get_wavelength()
        # value=round(value,3)
        self.lambEdit.setText(str(value))

    def update_rate(self):
        value=self.horizontalSlider_3.value()*0.01
        value=round(value,2)
        self.rateEdit.setText(str(value))
    
    def update_slider1(self):
        try:
            value=float(self.dedit.text())*1000
            self.horizontalSlider.setValue(value)
        except:
            return None
        
    def update_slider2(self):
        try:
            value=float(self.lambEdit.text())*10
            self.verticalSlider.setValue(value)
        except:
            return None
        
    def update_slider3(self):
        try:
            value=float(self.rateEdit.text())*100
            self.horizontalSlider_3.setValue(value)
        except:
            return None
        
    def load_image(self):
        self.img_path=QFileDialog.getOpenFileName(self,"choose an image","./","(*.jpg *.png *.jpeg)")[0]
        try:
            img_show=Image.open(self.img_path)
            self.img_size=img_show.width
            self.img_size_H=img_show.height
            
            self.image=preprocess_image(self.img_path,target_size=self.img_size,target_size_H=self.img_size_H).to(DEVICE)
            self.dis_diflb.clear()
            self.startbtn.setText("开始生成")
            self.startbtn.setEnabled(True)
        except:
            QMessageBox.critical(self,'Error',"Invalid Image Input")
            return None
        pixmap=QPixmap(self.img_path)
        pixmap=pixmap.scaled(351,351,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.dis_aperlb.setPixmap(pixmap)
    
    def resize_img(self):
        self.image=preprocess_image(self.img_path,self.img_size_H,self.img_size).to(DEVICE)
        pixmap=tensor_to_qpixmap_method2(self.image,target_size=(self.img_size,self.img_size_H))
        pixmap=pixmap.scaled(351,351,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.dis_aperlb.setPixmap(pixmap)

    def update_size(self):
        dialog=ResizeWind()
        dialog.show()
        dialog.widthEdit.setText(str(self.img_size))
        dialog.heightEdit.setText(str(self.img_size_H))
        if dialog.exec()==QDialog.Accepted:
            self.change_width(dialog.widthEdit.text())
            self.change_height(dialog.heightEdit.text())
            self.resize_img()

    def change_width(self,width):
        try:
            width=int(width)
            self.img_size=width
        except:
            a=QMessageBox.critical(self,'Error','Invalid Width')
        
    def change_height(self,height):
        try:
            height=int(height)
            self.img_size_H=height
        except:
            a=QMessageBox.critical(self,'Error','Invalid Height')

    def changeMode(self,index):
        self.startbtn.setText("开始生成")
        self.startbtn.setEnabled(True)
        if index==0:
            self.mode=0
            self.model=Fresnal(self.img_size,self.img_size_H,lambda_=self.lambda_,d=self.d,rate=self.rate).to(DEVICE)
            self.update_data()
            self.dedit.setText(str(round(self.d,3)))
            self.rateEdit.setText(str(round(self.rate*1000000,3)))
            self.lambEdit.setText(str(round(self.lambda_*1000000000,1)))
        if index==1:
            self.mode=1
            self.update_data()
            self.model=Fresnal_ex(self.img_size,lambda_=self.lambda_,d=self.d,rate=self.rate).to(DEVICE)
            self.dedit.setText(str(round(self.d,3)))
            self.rateEdit.setText(str(round(self.rate*1000000,3)))
            self.lambEdit.setText(str(round(self.lambda_*1000000000,1)))

    def update_data(self):
        try:
            d=float(self.dedit.text())
            rate=float(self.rateEdit.text())
            lamb=float(self.lambEdit.text())
            #dialog.update_images(d*10)
            #print(d,rate,lamb)
        except:
            a=QMessageBox.critical(self,'Error',"Invalid input!")
            return None
        if d<=0.001 or rate<=0 or lamb<=0:
            a=QMessageBox.critical(self,'Error',"Invalid Input!")
            return None
        self.d=d
        self.lamb=lamb*0.000000001
        self.rate=rate*0.000001

    def generate(self):
        if self.image==None:
            a=QMessageBox.critical(self,'Error',"No Aperture Input")
            return None
        self.update_data()
        if self.mode==0:
            self.model=Fresnal(self.img_size,self.img_size_H,lambda_=self.lambda_,d=self.d,rate=self.rate).to(DEVICE)
        if self.mode==1:
            self.model=Fresnal_ex(self.img_size,lambda_=self.lambda_,d=self.d,rate=self.rate).to(DEVICE)
        #print(self.image.size(),self.img_size,self.img_size_H)
        rec=self.model(self.image)
        self.rec=rec
        #torchvision.utils.save_image(rec,"test114.png")
        #print(rec.size())

        pixmap=tensor_to_qpixmap_method2(rec,(self.img_size,self.img_size_H))
        pixmap=pixmap.scaled(351,351,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.dis_diflb.setPixmap(pixmap)
        #self.startbtn.setText("生成完毕")
        #self.startbtn.setEnabled(False)
        self.save_btn.setEnabled(True)
    
    def save_results(self):
        if self.rec==None:
            return None
        #self.save_btn.setEnabled(False)
        dir_path=self.result_path
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir=os.path.join(dir_path,f"results_{current_time}")
        self.current_result_path=base_dir
        os.mkdir(base_dir)

        torchvision.utils.save_image(self.image,os.path.join(base_dir,self.aper_name),normalize=True)
        torchvision.utils.save_image(self.rec,os.path.join(base_dir,self.rec_name),normalize=False)

        
        
        with open(os.path.join(base_dir,self.init_name), 'w') as f:
            f.write(f"{self.d}\n")
            f.write(f"{self.lamb}\n")
            f.write(f"{self.rate}\n")
            f.write(f"{self.mode}\n")
        
app=QApplication([])
wind=MainWindow()
wind.show()
app.exec()