### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### Modified by Erik Härkönen, 2019
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

def main():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    # TIMBRER
    opt.dataroot = './datasets/timbre/'
    opt.name = 'timbrer'
    opt.resize_or_crop = 'none'
    opt.label_nc = 0
    opt.no_instance = True    
    opt.input_nc = 1
    opt.output_nc = 1
    opt.timbrer = True
    opt.how_many = 100 # how many results to generate
    opt.which_epoch = 'piano_guitar' #'harp_kalimba'
    opt.datasets = {
        'source': 'maestro_piano.npy',
        'target': 'maestro_guitar.npy'
    }
    
    opt.use_encoded_image = True # Adds GT to test set
    opt.use_features = False     # makes sure they aren't used in inference
    data_loader = CreateDataLoader(opt)

    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
                
        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
        
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                            opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = model.inference(data['label'], data['inst'], data['image'])
        
        def tensor_to_array(t):
            data = np.maximum(0.0, t.cpu().float().numpy())
            return data

        visuals = OrderedDict([
            ('input', tensor_to_array(data['label'][0])),
            ('output', tensor_to_array(generated.data[0])),
            ('reference', tensor_to_array(data['image'][0]))
        ])
        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

if __name__ == '__main__':
    main()
