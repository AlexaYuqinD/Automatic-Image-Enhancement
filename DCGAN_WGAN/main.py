from config import config
from load_dataset import *
from utils import *
from model import *

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as utils
from skimage.measure import compare_ssim
import time


class FeatureExtractor(nn.Sequential):
    """
    Feature extractor model
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def forward(self, x, feature_id):
        for idx, module in enumerate(self._modules):
            x = self._modules[module](x)
            if idx == feature_id:
                return x


def setup(checkpoint_path, sample_path):
    """
    :checkpoint_path: path of model checkpoints 
    :sample_path: path of sample outputs
    """
    if not os.path.exists(os.path.join(config.checkpoint_path, config.model_type)):
        os.makedirs(os.path.join(config.checkpoint_path, config.model_type))
    
    if not os.path.exists(os.path.join(config.sample_path, config.model_type)):    
        os.makedirs(os.path.join(config.sample_path, config.model_type))

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')

def get_feature_extractor(device):
    """
    Load pretrained vgg as a feature extractor
    """   
    vgg_temp = models.vgg19(pretrained=True).features
    model = FeatureExtractor()

    conv_counter = 1
    relu_counter = 1
    block_counter = 1

    for i, layer in enumerate(list(vgg_temp)):
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(block_counter) + '_' + str(conv_counter)
            conv_counter += 1
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu_' + str(block_counter) + '_' + str(relu_counter)
            relu_counter += 1
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            # TODO: try to use nn.AvgPool2d((2,2))
            name = 'pool_' + str(block_counter)
            relu_counter = conv_counter = 1
            block_counter += + 1
            model.add_module(name, layer)

    model.to(device)
    return model


def get_feature(model, img_tensor, feature_id, device):
    """
    Normalize and extract features from images
    
    :model: feature extractor
    :img_tensor: image with data type tensor
    :feature_id: layer id in vgg
    :device: cuda or cpu
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, config.channels, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, config.channels, 1, 1)
    img_normalized = (img_tensor - mean) / std
    feature = model(img_normalized, feature_id)
    return feature


def load_checkpoints(model):
    """
    Load model checkpoints to either test or continue training
    
    :model: image enhancer model
    """    
    print('Loading the model checkpoints from iter {}...'.format(config.resume_iter))
    checkpoint_path = os.path.join(config.checkpoint_path, config.model_type)

    gen_g_path = os.path.join(checkpoint_path, '{}-Gen_g.ckpt'.format(config.resume_iter))
    gen_f_path = os.path.join(checkpoint_path, '{}-Gen_f.ckpt'.format(config.resume_iter))
    model.gen_g.load_state_dict(torch.load(gen_g_path, map_location=lambda storage, loc: storage))
    model.gen_f.load_state_dict(torch.load(gen_f_path, map_location=lambda storage, loc: storage))

    if config.train:
        dis_c_path = os.path.join(checkpoint_path, '{}-Dis_c.ckpt'.format(config.resume_iter))
        dis_t_path = os.path.join(checkpoint_path, '{}-Dis_t.ckpt'.format(config.resume_iter))
        model.dis_c.load_state_dict(torch.load(dis_c_path, map_location=lambda storage, loc: storage))
        model.dis_t.load_state_dict(torch.load(dis_t_path, map_location=lambda storage, loc: storage))


def train(model, device):
    """
    train the model
    
    :model: image enhancer model
    :device: cuda or cpu
    """    
    extractor = get_feature_extractor(device)
    true_labels = torch.ones(config.batch_size, dtype=torch.long).to(device)
    false_labels = torch.zeros(config.batch_size, dtype=torch.long).to(device)
    settime = str(int(time.time()))
    logs = open('logs/' + settime + '.txt', "w+")
    logs.close()
    
    for idx in range(config.resume_iter, config.train_iters):
        train_original, train_style = load_train_dataset(config.data_path, config.batch_size,
                                                     config.height * config.width * config.channels)
        x = torch.from_numpy(train_original).float()
        y_real = torch.from_numpy(train_style).float()
        x = x.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)
        y_real = y_real.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)

        # --------------------------------------------------------------------------------------------------------------
        #                                                Train generators
        # --------------------------------------------------------------------------------------------------------------
        y_fake = model.gen_g(x)
        x_rec = model.gen_f(y_fake)

        # content loss
        feat_x = get_feature(extractor, x, config.feature_id, device)
        feat_x_rec = get_feature(extractor, x_rec, config.feature_id, device)
        loss_content = torch.pow(feat_x.detach() - feat_x_rec, 2).mean()

        # color loss
        # gaussian blur image for discriminator_c
        fake_blur = gaussian_blur(y_fake, config.kernel_size, config.sigma, config.channels, device)
        logits_fake_blur = model.dis_c(fake_blur)
        if config.model_type == 'DCGAN':
            loss_c = model.criterion(logits_fake_blur, true_labels)
        elif config.model_type == 'WGAN':
            loss_c = model.criterion(logits_fake_blur)
            
        # texture loss
        # gray-scale image for discriminator_t
        fake_gray = gray_scale(y_fake)
        logits_fake_gray = model.dis_t(fake_gray)
        if config.model_type == 'DCGAN':
            loss_t = model.criterion(logits_fake_gray, true_labels)
        elif config.model_type == 'WGAN':
            loss_t = model.criterion(logits_fake_gray)

        # total variation loss
        height_tv = torch.pow(y_fake[:, :, 1:, :] - y_fake[:, :, :config.height - 1, :], 2).mean()
        width_tv = torch.pow(y_fake[:, :, :, 1:] - y_fake[:, :, :, :config.width - 1], 2).mean()
        loss_tv = height_tv + width_tv

        # total generator loss
        gen_loss = loss_content + config.lambda_c * loss_c + config.lambda_t * loss_t + config.lambda_tv * loss_tv

        model.g_optimizer.zero_grad()
        model.f_optimizer.zero_grad()
        gen_loss.backward()
        model.g_optimizer.step()
        model.f_optimizer.step()
 

        # --------------------------------------------------------------------------------------------------------------
        #                                              Train discriminators
        # --------------------------------------------------------------------------------------------------------------
        y_fake = model.gen_g(x)
        
        # color loss
        fake_blur = gaussian_blur(y_fake, config.kernel_size, config.sigma, config.channels, device)
        real_blur = gaussian_blur(y_real, config.kernel_size, config.sigma, config.channels, device)
        logits_fake_blur = model.dis_c(fake_blur.detach())
        logits_real_blur = model.dis_c(real_blur.detach())
        if config.model_type == 'DCGAN':
            loss_dc = model.criterion(logits_real_blur, true_labels) + model.criterion(logits_fake_blur, false_labels)
        elif config.model_type == 'WGAN':
            loss_dc = model.criterion(logits_real_blur) + model.criterion(logits_fake_blur, false_labels)
            
        # texture loss
        fake_gray = gray_scale(y_fake)
        real_gray = gray_scale(y_real)
        logits_fake_gray = model.dis_t(fake_gray.detach())
        logits_real_gray = model.dis_t(real_gray.detach())
        if config.model_type == 'DCGAN':
            loss_dt = model.criterion(logits_real_gray, true_labels) + model.criterion(logits_fake_gray, false_labels)
        elif config.model_type == 'WGAN':
            loss_dt = model.criterion(logits_real_gray) - model.criterion(logits_fake_gray)
            
        # total discriminator loss
        dis_loss = config.lambda_c * loss_dc + config.lambda_t * loss_dt

        model.c_optimizer.zero_grad()
        model.t_optimizer.zero_grad()
        dis_loss.backward()
        model.c_optimizer.step()
        model.t_optimizer.step()
        
        # Add weight clamping for WGAN       
        if config.model_type == 'WGAN':
            for param in model.dis_c.parameter():
                param.data.clamp_(-config.clamp, config.clamp)
            for param in model.dis_t.parameter():
                param.data.clamp_(-config.clamp, config.clamp)        

        print('Iteration : {}/{}, Gen_loss : {:.4f}, Dis_loss : {:.4f}'.format(
            idx + 1, config.train_iters, gen_loss.data, dis_loss.data))
        print('Loss_content : {:.4f}, Loss_c : {:.4f}, Loss_t : {:.4f}, Loss_tv: {:.4f}'.format(
            loss_content.data, loss_c.data, loss_t.data, loss_tv.data))
        print('Loss_dc : {:.4f}, Loss_dt : {:.4f}'.format(loss_dc.data, loss_dt.data))
        
        if (idx + 1) % 50 == 0:
            log1 = 'Iteration : {}/{}, Gen_loss : {:.4f}, Dis_loss : {:.4f}'.format(
                idx + 1, config.train_iters, gen_loss.data, dis_loss.data)
            log2 = 'Loss_content : {:.4f}, Loss_c : {:.4f}, Loss_t : {:.4f}, Loss_tv: {:.4f}'.format(
                loss_content.data, loss_c.data, loss_t.data, loss_tv.data)
            log3 = 'Loss_dc : {:.4f}, Loss_dt : {:.4f}'.format(loss_dc.data, loss_dt.data)
            logs = open('logs/' + settime + '.txt', "a")
            logs.write(log1)
            logs.write('\n')
            logs.write(log2)
            logs.write('\n')
            logs.write(log3)
            logs.write('\n')
            logs.close()

        if (idx + 1) % 1000 == 0:
            sample_path = os.path.join(config.sample_path, config.model_type)
            checkpoint_path = os.path.join(config.checkpoint_path, config.model_type)

            utils.save_image(x, os.path.join(sample_path, '{}-x.jpg'.format(idx + 1)))
            utils.save_image(x_rec, os.path.join(sample_path, '{}-x_rec.jpg'.format(idx + 1)))
            utils.save_image(y_fake, os.path.join(sample_path, '{}-y_fake.jpg'.format(idx + 1)))
            utils.save_image(y_real, os.path.join(sample_path, '{}-y_real.jpg'.format(idx + 1)))
            utils.save_image(fake_blur, os.path.join(sample_path, '{}-fake_blur.jpg'.format(idx + 1)))
            utils.save_image(real_blur, os.path.join(sample_path, '{}-real_blur.jpg'.format(idx + 1)))
            utils.save_image(fake_gray, os.path.join(sample_path, '{}-fake_gray.jpg'.format(idx + 1)))
            utils.save_image(real_gray, os.path.join(sample_path, '{}-real_gray.jpg'.format(idx + 1)))

            torch.save(model.gen_g.state_dict(), os.path.join(checkpoint_path, '{}-Gen_g.ckpt'.format(idx + 1)))
            torch.save(model.gen_f.state_dict(), os.path.join(checkpoint_path, '{}-Gen_f.ckpt'.format(idx + 1)))
            torch.save(model.dis_c.state_dict(), os.path.join(checkpoint_path, '{}-Dis_c.ckpt'.format(idx + 1)))
            torch.save(model.dis_t.state_dict(), os.path.join(checkpoint_path, '{}-Dis_t.ckpt'.format(idx + 1)))
            print('Saved intermediate images and model checkpoints.')


def test_patches(model, device):
    """
    Test the trained model with patches
    
    :model: image enhancer model
    :device: cuda or cpu
    """   
    test_path = config.data_path + '/test/original/'
    test_image_num = len([name for name in os.listdir(test_path)
                         if os.path.isfile(os.path.join(test_path, name))]) // config.batch_size * config.batch_size

    score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar = 0.0, 0.0, 0.0, 0.0
    for start in range(0, test_image_num, config.batch_size):
        end = min(start + config.batch_size, test_image_num)
        test_original, test_style = load_test_dataset_patches(config.data_path, start, end,
                                                  config.height * config.width * config.channels)
        x = torch.from_numpy(test_original).float()
        y_real = torch.from_numpy(test_style).float()
        x = x.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)
        y_real = y_real.view(-1, config.height, config.width, config.channels).permute(0, 3, 1, 2).to(device)

        y_fake = model.gen_g(x)

        # Calculate PSNR & SSIM scores
        score_psnr += psnr(y_fake, y_real) * config.batch_size

        y_fake_np = y_fake.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_real_np = y_real.cpu().numpy().transpose(0, 2, 3, 1)
        temp_ssim, _ = compare_ssim(y_fake_np, y_real_np, multichannel=True, gaussian_weights=True, full=True)
        score_ssim_skimage += (temp_ssim * config.batch_size)

        temp_ssim, _ = ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5)
        score_ssim_minstar += temp_ssim * config.batch_size

        score_msssim_minstar += multi_scale_ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5) * config.batch_size
        print('PSNR & SSIM scores of {} images are calculated.'.format(end))

    score_psnr /= test_image_num
    score_ssim_skimage /= test_image_num
    score_ssim_minstar /= test_image_num
    score_msssim_minstar /= test_image_num
    print('PSNR : {:.4f}, SSIM_skimage : {:.4f}, SSIM_minstar : {:.4f}, SSIM_msssim: {:.4f}'.format(
        score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar))


def test(model, device):
    """
    test the trained model with full images
    
    :model: image enhancer model
    :device: cuda or cpu
    """   
    test_path = '../data/full/original/'
    test_image_num = len([name for name in os.listdir(test_path)
                         if os.path.isfile(os.path.join(test_path, name))]) // config.batch_size * config.batch_size

    score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar = 0.0, 0.0, 0.0, 0.0
    for ind in range(0, test_image_num):
        test_original, test_style, image_height, image_width = load_test_dataset(ind)
        x = torch.from_numpy(test_original).float()
        y_real = torch.from_numpy(test_style).float()
        x = x.view(-1, image_height, image_width, config.channels).permute(0, 3, 1, 2).to(device)
        y_real = y_real.view(-1, image_height, image_width, config.channels).permute(0, 3, 1, 2).to(device)

        y_fake = model.gen_g(x)

        # Calculate PSNR & SSIM scores
        score_psnr += psnr(y_fake, y_real)

        y_fake_np = y_fake.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_real_np = y_real.cpu().numpy().transpose(0, 2, 3, 1)
        temp_ssim, _ = compare_ssim(y_fake_np, y_real_np, multichannel=True, gaussian_weights=True, full=True)
        score_ssim_skimage += temp_ssim

        temp_ssim, _ = ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5)
        score_ssim_minstar += temp_ssim

        score_msssim_minstar += multi_scale_ssim(y_fake, y_real, kernel_size=11, kernel_sigma=1.5)
        print('PSNR & SSIM scores of {} images are calculated.'.format(end))

    score_psnr /= test_image_num
    score_ssim_skimage /= test_image_num
    score_ssim_minstar /= test_image_num
    score_msssim_minstar /= test_image_num
    print('PSNR : {:.4f}, SSIM_skimage : {:.4f}, SSIM_minstar : {:.4f}, SSIM_msssim: {:.4f}'.format(
        score_psnr, score_ssim_skimage, score_ssim_minstar, score_msssim_minstar))



def main():
    setup(config.checkpoint_path, config.sample_path)
    
    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    model = Enhancer(config, device)
    if config.resume_iter != 0:
        load_checkpoints(model)

    if config.train:
        train(model, device)
    elif config.test:
        test(model, device)
    else:
        test_patches(model, device)


if __name__ == '__main__':
    main()
