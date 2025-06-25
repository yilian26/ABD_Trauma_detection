import gc
import os
import torch
import time
from datetime import datetime
import numpy as np
import torch.nn as nn
from skimage.transform import resize
from utils.visualizer import (
    plot_heatmap_one_picture, 
    plot_heatmap_detail,
    plot_vedio,
    process_plot_multiprocess,
)

'''
參考https://github.com/yizt/Grad-CAM.pytorch/tree/master/detection
'''

def get_last_encoder_conv_name(net):
    """
    Get the name of the last convolutional layer in the encoder part of a network.
    This function assumes that decoder layers follow the encoder layers and are named with 'decoder'.
    
    :param net: The network from which to get the last encoder convolutional layer name.
    :return: The name of the last convolutional layer in the encoder.
    """
    last_conv_name = None
    for name, module in net.named_modules():
        if "decoder" in name:  # Assumes that 'decoder' in the name indicates the start of decoder layers.
            break
        if isinstance(module, nn.Conv3d):
            last_conv_name = name
    return last_conv_name

def get_organ_list(class_type: str):
    class_type_to_organs = {
        "all": ["inj_solid"],
        "multiple": ["Liver", "Spleen", "Kidney"],
        "liver": ["Liver"],
        "spleen": ["Spleen"],
        "kidney": ["Kidney"],
    }

    organ_list = class_type_to_organs.get(class_type.lower())
    if organ_list is None:
        raise ValueError(f"Unsupported class_type: {class_type}")
    return organ_list


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, device):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.device = device
        # 使模型能夠計算gradient
        self.net.eval()
        # 儲存feature和gradient的list(hook資料型態)
        self.handlers = []
        # 將 feature與gradient 取出
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple,  input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,  长度为1
        :return:
        """
        self.gradient = output_grad[0]

    # 利用hook取指定層的feature和gradient
    def _register_hook(self):
        if self.layer_name == 'multiply':
            self.handlers.append(self.net.multiply.register_forward_hook(self._get_features_hook))
            self.handlers.append(self.net.multiply.register_backward_hook(self._get_grads_hook))
        else:
            for (name, module) in self.net.named_modules():
                if name == self.layer_name:
                    #print("OK")
                    # forward取feature
                    self.handlers.append(module.register_forward_hook(self._get_features_hook))
                    # backward取gradient
                    self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                #else:
                    #print("Nothing to do")
    # 每次計算完後就把結果刪除，避免memory不夠
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, bbox=False, index_sel=None, normalize=True):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index_sel: list for label
        :return:  a list of each batch heatmap
        """
        # 將模型gradinet歸零 (應該可以不用)
        self.net.zero_grad()
        # inference取得output
        if torch.is_tensor(bbox):
            output = self.net(inputs, bbox)
        else:
            output = self.net(inputs) # [1,num_classes]
        if isinstance(output, tuple):
            output = AngleLoss_predict()(output)

        # 針對output準備取得feature與gradient    
        heatmap_list = []
        #print(f"output shape: {output.shape}")
        for i in range(output.shape[0]):
            # 如果沒有指定取得的標籤，預設則使用預測的標籤
            if  index_sel == None:
                index = np.argmax(output[i,:].cpu().data.numpy())
                print(f"predict:{index}")  
            else:
                index = int(index_sel[i])
                print(f"predict:{index}")
          
            # backward
            target = output[i][index]
            # 將backward的結果保留，才能取得gradient
            target.backward(retain_graph=True)
            #取得gradient和feature
            gradient = self.gradient[i].cpu().detach().data.numpy()  # [C,H,W,D]
            weight = np.mean(gradient, axis=(1, 2, 3))  # [C]

            feature = self.feature[i].cpu().detach().data.numpy()  # [C,H,W,D]
            
            # 計算grad cam方法
            # np.newaxis 增加維度的方法
            cam = feature * weight[:, np.newaxis, np.newaxis, np.newaxis]  # [C,H,W,D] feature map 與 weight相乘
            cam = np.sum(cam, axis=0)  # [H,W,D] 
            cam = np.maximum(cam, 0)  # ReLU
            # normalize 
            if normalize:
                heatmap = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                heatmap = cam
            heatmap = resize(heatmap,inputs.shape[2:5])
            heatmap = heatmap.transpose( 1, 2,0)
            print(heatmap.shape)
#             heatmap = heatmap.transpose( 1, 2,0)
            heatmap_list.append(heatmap)

        return heatmap_list


class GradCAM_seg(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, device):
        self.net = net.module if isinstance(net, torch.nn.DataParallel) else net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.device = device
        # 使模型能夠計算gradient
        self.net.eval()
        # 儲存feature和gradient的list(hook資料型態)
        self.handlers = []
        # 將 feature與gradient 取出
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple,  input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,  长度为1
        :return:
        """
        self.gradient = output_grad[0]

    # 利用hook取指定層的feature和gradient
    def _register_hook(self):
        if self.layer_name == 'multiply':
            self.handlers.append(self.net.multiply.register_forward_hook(self._get_features_hook))
            self.handlers.append(self.net.multiply.register_backward_hook(self._get_grads_hook))
        else:
            for (name, module) in self.net.named_modules():
                if name == self.layer_name:
                    #print("OK")
                    # forward取feature
                    self.handlers.append(module.register_forward_hook(self._get_features_hook))
                    # backward取gradient
                    self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                #else:
                    #print("Nothing to do")
    # 每次計算完後就把結果刪除，避免memory不夠
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, bbox=False, index_sel=None, normalize=True):
        """
        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index_sel: list for label
        :return:  a list of each batch heatmap
        """
        # 將模型gradinet歸零 (應該可以不用)
        self.net.zero_grad()
        # inference取得output
        if torch.is_tensor(bbox):
            output = self.net(inputs, bbox)
        else:
            output, mask = self.net(inputs) # [1,num_classes]

        # 針對output準備取得feature與gradient 
        if mask.shape[1] > 1:
            heatmap_list = {organ: [] for organ in ["Liver", "Spleen", "Kidney"]}
            organ_list = ["Liver","Spleen","Kidney"]
            print(organ_list, len(organ_list))
        else:
            heatmap_list = {organ: [] for organ in ["organ"]}
            organ_list = ["organ"]
            print(organ_list, len(organ_list))
        #print(f"output shape: {output.shape}")
        for organ_idx, organ in enumerate(organ_list):
            for i in range(output.shape[0]):
                # 如果沒有指定取得的標籤，預設則使用預測的標籤
                if  index_sel == None:
                    if len(organ_list) ==1:
                        index = np.argmax(output[i,:].cpu().data.numpy())
                    else:
                        index = np.argmax(output[i, organ_idx,:].cpu().data.numpy())
                    print(f"predict:{index}")  
                else:
                    index = int(index_sel[i])
                    print(f"predict:{index}")

                
                if len(organ_list) ==1:
                    target = output[i, index]
                else:
                    target = output[i, organ_idx, index]
                # 將backward的結果保留，才能取得gradient
                target.backward(retain_graph=True)
                #取得gradient和feature
                gradient = self.gradient[i].cpu().detach().data.numpy()  # [C,H,W,D]
                weight = np.mean(gradient, axis=(1, 2, 3))  # [C]

                feature = self.feature[i].cpu().detach().data.numpy()  # [C,H,W,D]

                # 計算grad cam方法
                # np.newaxis 增加維度的方法
                cam = feature * weight[:, np.newaxis, np.newaxis, np.newaxis]  # [C,H,W,D] feature map 與 weight相乘
                cam = np.sum(cam, axis=0)  # [H,W,D] 
                cam = np.maximum(cam, 0)  # ReLU
                # normalize 
                if normalize:
                    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
                else:
                    heatmap = cam
                print("before resize", heatmap.shape)
                heatmap = resize(heatmap,inputs.shape[2:5])
                heatmap = heatmap.transpose( 1, 2,0)
                print(heatmap.shape)
#             heatmap = heatmap.transpose( 1, 2,0)
                heatmap_list[organ].append(heatmap)
                del feature, gradient, cam, heatmap, target
                gc.collect()
                torch.cuda.empty_cache()

        return heatmap_list


def get_gradcam_instance(model, architecture, device):
    if architecture == "unet":
        return GradCAM_seg(model, "layer4.2.conv3", device)
    else:
        layer_name = get_last_encoder_conv_name(model)
        return GradCAM(model, layer_name, device)


def run_gradcam_batch(model, test_images, test_labels, architecture, class_type, device, Label):
    grad_cam = get_gradcam_instance(model, architecture, device)

    try:
        if Label == "True":
            result_list = grad_cam(test_images, index_sel=test_labels)
        else:
            result_list = grad_cam(test_images)
    finally:
        grad_cam.remove_handlers()
    
    del grad_cam
    torch.cuda.empty_cache()
    gc.collect()
    
    return result_list


def save_gradcam_results(result_list, testdata, test_df, organ_list, k, dir_path, heatmap_type):
    test_images = testdata["image"].to("cpu")  # CPU操作夠用
    file_id_list = test_df.file_id

    for organ_idx, organ in enumerate(organ_list):
        for i in range(len(result_list[organ])):
            idx = k + i
            file_path = file_id_list[idx]
            image = test_images[i, 0].numpy()
            heatmap_total = result_list[organ][i]

            # 判斷是否為正樣本
            if organ == "all":
                pos_label = test_df["inj_solid"][idx] == 1
            elif organ == "Liver":
                pos_label = test_df["liver_label"][idx] == 1
            elif organ == "Spleen":
                pos_label = test_df["spleen_label"][idx] == 1
            elif organ == "Kidney":
                pos_label = test_df["kidney_label"][idx] == 1
            else:
                pos_label = False

            # 路徑組合
            pos_path = f"{dir_path}/POS/{organ}/{file_path}"
            neg_path = f"{dir_path}/NEG/{organ}/{file_path}"
            pos_total = f"{dir_path}/POS_total/{organ}/{file_path}_total"
            neg_total = f"{dir_path}/NEG_total/{organ}/{file_path}_total"

            final_path = pos_path if pos_label else neg_path
            total_final_path = pos_total if pos_label else neg_total

            os.makedirs(final_path, exist_ok=True)
            os.makedirs(total_final_path, exist_ok=True)

            print(f"Read file in line {idx} by {organ}")
            print("channel_first", image.shape)

            # 圖片處理與儲存
            if heatmap_type in ["detail", "all"]:
                for j in range(image.shape[-1]):
                    plot_heatmap_detail(
                        heatmap_total[:, :, j],
                        image[:, :, j],
                        f"{final_path}/{j:03}.png",
                    )
                plot_vedio(final_path)

            if heatmap_type in ["one_picture", "all"]:
                plot_heatmap_one_picture(
                    heatmap_total, image, f"{total_final_path}/total_view.png"
                )

            print(f"{file_path} is already done!", flush=True)

    del test_images, result_list
    torch.cuda.empty_cache()
    gc.collect()


def CAM_plot(conf, model, test_loader, test_df, class_type, device, first_idx, heatmap_type, Label, dir_path):
    size = conf.augmentation.size
    architecture = conf.data_setting.architecture
    k = first_idx
    organ_list = get_organ_list(class_type)

    for testdata in test_loader:
        # Step 1: Run GradCAM
        test_images = testdata["image"].permute(0, 1, 4, 2, 3).to(device)
        test_labels = testdata["label"].to(device)

        start_time = time.time()
        result_list = run_gradcam_batch(model, test_images, test_labels, architecture, class_type, device, Label)
        print("Time taken for GradCAM: ", time.time() - start_time)

        # Step 2: Save Results
        start_time = time.time()
        save_gradcam_results(result_list, testdata, test_df, organ_list, k, dir_path, heatmap_type)
        print("Time taken for plotting: ", time.time() - start_time)

        # Index advance
        k += len(testdata["image"])

        # 清理顯式變數
        del test_images, test_labels, testdata
        torch.cuda.empty_cache()
        gc.collect()

