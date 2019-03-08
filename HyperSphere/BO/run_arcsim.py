import argparse
import os.path
import pickle
import sys
import time
import logging

import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

# General
from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates, optimization_init_points, deepcopy_inference, N_INIT
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.test_functions.benchmarks import *
from HyperSphere.test_functions.mnist_weight import mnist_weight

# Kernels
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel

# Inferences
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_satellite import ShadowInference as satellite_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.BO.shadow_inference.inference_sphere_origin_satellite import ShadowInference as both_ShadowInference

################################################################################
################################################################################

EXPERIMENT_DIR = '/home/tomrunia/experiments/2019_WindSpeed/sim_refinement'

################################################################################
################################################################################


def arcsim_blackbox_optimization(config, distance_func, target_video, arcsim_conf_scaler,
                                 geometry=None, n_eval=200, path=None, 
                                 func=None, ndim=None, boundary=False, ard=False, 
                                 origin=False, logger=None):

    start_time = time.time()

    if geometry is None or geometry != 'cube':
        raise ValueError('Please set geometry=cube for ArcSim optimization.')

    if path is not None:
        raise NotImplementedError('Restoring from path not supported.')

    if not os.path.isdir(EXPERIMENT_DIR):
        raise NotADirectoryError('EXPERIMENT_DIR variable is not properly assigned. Please check it.')

    if not isinstance(distance_func, nn.Module):
        raise ValueError('Distance function must be an instance of nn.Module')
    
    if not isinstance(target_video, torch.Tensor):
        raise ValueError('Target video must be an instance of torch.Tensor')

    if origin:
        raise ValueError('Origin must be set to False.')

    if (func.dim == 0 and ndim is not None):
        assert func.dim == ndim

    assert (path is None) != (func is None)

    if logger is None:
        logger = logging.getLogger()

    ################################################################################
    # Define the Model

    kernel = Matern52(ndim=ndim, ard=ard)
    model = GPRegression(kernel=kernel)
    inference_method = satellite_ShadowInference if boundary else Inference
    bnd = (-1, 1)

    ################################################################################
    # Prepare directories 

    dir_list = [elm for elm in os.listdir(EXPERIMENT_DIR) if os.path.isdir(os.path.join(EXPERIMENT_DIR, elm))]

    exp_conf_str = geometry
    exp_conf_str += ('_ARD' if ard else '') + ('_boundary' if boundary else '')

    folder_name = '{timeprefix}_{func}_dims{dims}_{conf}'.format(
        timeprefix=time.strftime('%Y%m%d_%H%M%S'),
        func=func.__name__,
        dims=ndim,
        conf=exp_conf_str)

    logfile_dir = os.path.join(config.output_dir, 'bo_logs')
    os.makedirs(logfile_dir)

    model_filename = os.path.join(config.output_dir, 'model.pt')
    data_config_filename = os.path.join(config.output_dir, 'data_config.pkl')

    ################################################################################

    x_input = Variable(torch.stack([torch.zeros(ndim), torch.FloatTensor(ndim).uniform_(-1, 1)]))
    n_init_eval = x_input.size(0)
    output = Variable(torch.zeros(n_init_eval, 1))

    for i in range(n_init_eval):
        # Actual function call to arcsim_refinement_function
        output[i] = func(x_input[i], config, distance_func, target_video, arcsim_conf_scaler, logger)

    time_list = [time.time()] * n_init_eval
    elapse_list = [0] * n_init_eval
    pred_mean_list = [0] * n_init_eval
    pred_std_list = [0] * n_init_eval
    pred_var_list = [0] * n_init_eval
    pred_stdmax_list = [1] * n_init_eval
    pred_varmax_list = [1] * n_init_eval
    reference_list = [output.data.squeeze()[0]] * n_init_eval
    refind_list = [1] * n_init_eval
    dist_to_ref_list = [0] * n_init_eval
    sample_info_list = [(10, 0, 10)] * n_init_eval

    # Inference and Sampling
    inference = inference_method((x_input, output), model)
    inference.init_parameters()
    inference.sampling(n_sample=1, n_burnin=99, n_thin=1)

    ############################################################################
    # Saving the Model

    ignored_variable_names = [
        'n_eval', 'path', 'i', 'key', 'value', 'logfile_dir', 'n_init_eval',
        'data_config_file', 'dir_list', 'folder_name', 'model_filename', 
        'data_config_filename', 'kernel', 'model', 'inference', 'parallel', 'pool']

    stored_variable_names = set(locals().keys()).difference(set(ignored_variable_names))

    if path is None:
        torch.save(model, model_filename)
        stored_variable = dict()
        for key in stored_variable_names:
            stored_variable[key] = locals()[key]
        f = open(data_config_filename, 'wb')
        pickle.dump(stored_variable, f)
        f.close()

    print(('Experiment based on data in %s' % os.path.split(model_filename)[0]))

    ############################################################################

    for curr_eval in range(n_eval):

        print('#'*60)
        print('Current evaluation: {}/{}'.format(curr_eval+1, n_eval))
        print('#'*60)

        start_time = time.time()
        logfile = open(os.path.join(logfile_dir, str(x_input.size(0) + 1).zfill(4) + '.out'), 'w')
        inference = inference_method((x_input, output), model)

        reference, ref_ind = torch.min(output, 0)
        reference = reference.item()
        gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
        inferences = deepcopy_inference(inference, gp_hyper_params)

        x0_cand = optimization_candidates(x_input, output, -1, 1)
        x0, sample_info = optimization_init_points(x0_cand, reference=reference, inferences=inferences)
        next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = suggest(x0=x0, reference=reference, inferences=inferences, bounds=bnd, pool=None)

        time_list.append(time.time())
        elapse_list.append(time_list[-1] - time_list[-2])
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())
        pred_stdmax_list.append(pred_stdmax.item())
        pred_varmax_list.append(pred_varmax.item())
        reference_list.append(reference)
        refind_list.append(ref_ind.item() + 1)
        dist_to_ref_list.append(torch.sum((next_x_point - x_input[ref_ind]) ** 2) ** 0.5)
        sample_info_list.append(sample_info)

        x_input = torch.cat([x_input, next_x_point], 0)

        # Actual function call to arcsim_refinement_function
        func_output = func(x_input[-1], config, distance_func, target_video, arcsim_conf_scaler, logger)
        func_output = func_output.unsqueeze(0).resize(1, 1)
        output = torch.cat([output, func_output])

        min_ind = torch.min(output, 0)[1]
        min_loc = x_input[min_ind]
        min_val = output[min_ind]
        dist_to_suggest = torch.sum((x_input - x_input[-1]).data ** 2, 1) ** 0.5
        dist_to_min = torch.sum((x_input - min_loc).data ** 2, 1) ** 0.5
        out_of_box = torch.sum((torch.abs(x_input.data) > 1), 1)

        print('')
        for i in range(x_input.size(0)):
            
            min_str = '<== MINIMUM' if i == min_ind.item() else ''
            print("{datestring} | Step {step} | output: {output:.4f} | R: {R:.4f} | OutOfBox: {out_of_box:.4f} {min_str}".format(
                datestring=datetime.now().strftime("%A %H:%M"), 
                step=i+1, 
                output=output.squeeze()[i],
                R=torch.sum(x_input.data[i]**2)**0.5,
                out_of_box=out_of_box[i],
                min_str=min_str)
            )

            #time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[i])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[i])) + ')  '

            # data_str = ('%3d-th : %+12.4f(R:%8.4f[%4d]/ref:[%3d]%8.4f), sample([%2d] best:%2d/worst:%2d), '
            #             'mean : %+.4E, std : %.4E(%5.4f), var : %.4E(%5.4f), '
            #             '2ownMIN : %8.4f, 2curMIN : %8.4f, 2new : %8.4f' %
            #             (i + 1, output.data.squeeze()[i], torch.sum(x_input.data[i] ** 2) ** 0.5, out_of_box[i], refind_list[i], reference_list[i],
            #              sample_info_list[i][2], sample_info_list[i][0], sample_info_list[i][1],
            #              pred_mean_list[i], pred_std_list[i], pred_std_list[i] / pred_stdmax_list[i], pred_var_list[i], pred_var_list[i] / pred_varmax_list[i],
            #              dist_to_ref_list[i], dist_to_min[i], dist_to_suggest[i]))
            # min_str = '  <========= MIN' if i == min_ind.item() else ''
            # print((time_str + data_str + min_str))
            # logfile.writelines(time_str + data_str + min_str + '\n')

        logfile.close()

        torch.save(model, model_filename)
        stored_variable = dict()
        for key in stored_variable_names:
            stored_variable[key] = locals()[key]
        f = open(data_config_filename, 'wb')
        pickle.dump(stored_variable, f)
        f.close()

    duration = time.time() - start_time
    time_delta = datetime.timedelta(seconds=duration)
    hours   = int(str(time_delta).split(':')[0])
    minutes = int(str(time_delta).split(':')[1])
    seconds = int(str(time_delta).split(':')[2][0:2])

    print('Experiment based on data in %s' % os.path.split(model_filename)[0])
    print('Experiment with {} evaluations done in {} hours, {} minutes and {} seconds.'.format(n_eval, hours, minutes, seconds))
    
    return os.path.split(model_filename)[0]

################################################################################
################################################################################

# if __name__ == '__main__':

#     config = dict()
    
#     parser = argparse.ArgumentParser(description='ArcSim Optimization')
#     parser.add_argument('--geometry', dest='geometry', default='cube')
#     parser.add_argument('--num_eval', type=int, default=10)
#     parser.add_argument('--num_dims', type=int, default=None)
#     parser.add_argument('--ard', dest='ard', action='store_true', default=False)

#     config = parser.parse_args()
#     config.path = None

#     optimization_func = branin
#     optimization_dims = branin.dim if config.num_dims is None else config.num_dims

#     arcsim_blackbox_optimization(
#         geometry=config.geometry,
#         n_eval=config.num_eval,
#         path=config.path,
#         func=optimization_func,
#         ndim=optimization_dims,
#         boundary=False,
#         ard=config.ard,
#         origin=False,
#     )
