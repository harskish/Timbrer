### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### Modified by Erik Härkönen, 2019
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import math
def lcm(a,b): return abs(a * b)/math.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

def main():
    opt = TrainOptions().parse()

    # TEST
    opt.dataroot = 'D:/datasets/timbrer/torch'
    opt.save_epoch_freq = 2 # save checkpoint every N epochs
    opt.niter = 5       # 5 epochs at full LR
    opt.niter_decay = 5 # 5 epochs of linear LR ramp down
    opt.resize_or_crop = 'none'
    opt.label_nc = 0
    opt.no_instance = True
    opt.timbrer = True
    opt.input_nc = 1
    opt.output_nc = 1
    
    
    opt.noise_src = True; opt.noise_trg = False; opt.name = 'piano_guitar_denoise' # Denoise
    #opt.noise_src = True; opt.noise_trg = True; opt.name = 'piano_guitar_aug' # Augment (ignore noise)
    
    opt.datasets = {
        'source': 'torch_logmel_maestro_piano.npy',
        'target': 'torch_logmel_maestro_guitar.npy'
    }

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    if opt.fp16:    
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
            else:
                loss_G.backward()          
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step()        

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input', np.maximum(0.0, data['label'][0].cpu().float().numpy())),
                                    ('synthesized', np.maximum(0.0, generated.data[0].cpu().float().numpy())),
                                    ('reference', np.maximum(0.0, data['image'][0].cpu().float().numpy()))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()

# Windows + multiprocessing requires this guard
if __name__ == '__main__':
    main()