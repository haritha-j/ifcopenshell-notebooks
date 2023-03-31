# use gradient descent to fine tune parameters using chamfer loss

import time
import torch
import copy
from tqdm import tqdm_notebook as tqdm

from src.chamfer import get_chamfer_loss_tensor
from src.visualisation import visualize_predictions
from src.utils import scale_preds


# single thread, adam optimiser
def chamfer_fine_tune_single(n_iter, step_size, preds, cloud, cat, blueprint):
    print(cloud.shape)
    cuda = torch.device('cuda')
    #cloud = cloud.transpose(1,0)
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = scale_preds(preds_copy.tolist(), cat)
    preds_t = torch.tensor([preds], requires_grad=True, device=cuda)
    cloud_t = torch.tensor([cloud], device=cuda)

    optimiser = torch.optim.Adam([preds_t], lr=step_size)

    for i in tqdm(range (n_iter)):
        t1 = time.perf_counter()

        optimiser.zero_grad()
        chamfer_loss = get_chamfer_loss_tensor(preds_t, cloud_t, cat)
        t2 = time.perf_counter()
                # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        chamfer_loss.backward()
        optimiser.step()
        
        t3 = time.perf_counter()
        #print("chamf", t2-t1, "grad", t3-t2)

        print(i, "loss", chamfer_loss.item(), "preds", preds_t.detach().cpu().numpy())
        
    # visualise
    modified_preds = preds_t.detach().cpu().numpy()[0].tolist()
    scaled_preds = scale_preds(modified_preds, cat)
    v_orignal, _ = visualize_predictions([cloud.transpose(1,0).tolist()], cat, [scaled_original_preds], blueprint, visualize=True)
    v_modified, _ = visualize_predictions([cloud.transpose(1,0).tolist()], cat, [scaled_preds], blueprint, visualize=True)
    
    return [v_orignal,v_modified], chamfer_loss.item()


# multi element Adam with single loss
def chamfer_fine_tune(n_iter, step_size, preds, cloud, cat, blueprint, alpha=1.0, visualise=True):
    # prepare data on gpu and setup optimiser
    cuda = torch.device('cuda')
    preds_copy = copy.deepcopy(preds)
    scaled_original_preds = [scale_preds(pc.tolist(), cat) for pc in preds_copy]
    preds_t = torch.tensor(preds, requires_grad=True, device=cuda)
    cloud_t = torch.tensor(cloud, device=cuda)
    optimiser = torch.optim.Adam([preds_t], lr=step_size)

    # check initial loss
    chamfer_loss, gen_cloud = get_chamfer_loss_tensor(preds_t, cloud_t, cat, reduce=False, return_cloud=True)
    #print("intial loss", chamfer_loss)   

    # iterative refinement with adam
    for i in tqdm(range (n_iter)):
        optimiser.zero_grad()
        chamfer_loss = get_chamfer_loss_tensor(preds_t, cloud_t, cat, alpha=alpha)
        chamfer_loss.backward()
        optimiser.step()

        print(i, "loss", chamfer_loss.detach().cpu().numpy())#, "preds", preds_t)
        
    # check final loss
    chamfer_loss, gen_cloud_mod = get_chamfer_loss_tensor(preds_t, cloud_t, cat, reduce=False, return_cloud=True)
    #print("final loss", chamfer_loss)
    modified_preds = preds_t.detach().cpu().numpy()
    
    # visualise
    if visualise:
        error_count = 0
        scaled_preds = [scale_preds(p.tolist(), cat) for p in modified_preds]
        visualisers = []
        cloud_visualisers = []
        
        gen_cloud = gen_cloud.detach().cpu().numpy()
        gen_cloud_mod = gen_cloud_mod.detach().cpu().numpy()          

        for i, p in enumerate(scaled_preds):
            try:
                v_orignal, _ = visualize_predictions([cloud[i].transpose(1,0).tolist()], cat, [scaled_original_preds[i]], blueprint, visualize=True)
                v_modified, _ = visualize_predictions([None, None, cloud[i].transpose(1,0).tolist()], cat, [scaled_preds[i]], blueprint, visualize=True)
                visualisers.append(v_orignal)
                visualisers.append(v_modified)
            except:
                error_count += 1
                v_orignal, _ = visualize_predictions([cloud[i].transpose(1,0).tolist(), gen_cloud[i].tolist()], cat, [], blueprint, visualize=True)
                v_modified, _ = visualize_predictions([None, gen_cloud_mod[i].tolist(), cloud[i].transpose(1,0).tolist()], cat, [], blueprint, visualize=True)
                cloud_visualisers.append(v_orignal)
                cloud_visualisers.append(v_modified)
    
#         return v_orignal,v_modified, modified_preds
        print("errors ", error_count)
        return cloud_visualisers, visualisers, modified_preds
    else:
        return modified_preds