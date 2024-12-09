import json
import logging
import math
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from train.open_clip_train.distributed import is_master
from train.open_clip_train.zero_shot import zero_shot_eval
from train.open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def calc_acc(ref_logits, label):
    correct_samples = 0
    pred = torch.ones_like(label)
    pred[ref_logits > 0] = 0
    correct_samples += (pred == label).sum()
    return correct_samples/ref_logits.shape[0]
        
        
def contrastive_loss(logit_scale, ref, candidates, label, device):
    ref_logits = ref @ candidates.T  
    pred = torch.argmax(ref_logits, dim=1)
    correct_samples = (pred == label).sum()
    acc = correct_samples/ref_logits.shape[0]
    return F.cross_entropy(ref_logits, label), acc 

def binary_loss(logit_scale, ref, candidates, label, device):
    raise NotImplemented
    candidates = candidates.reshape(-1, 2, *candidates.shape[1:])
    ref_logits = torch.stack((torch.sum(ref * candidates[:,0], dim=1), torch.sum(ref * candidates[:,1], dim=1)), dim=1)
    return F.cross_entropy(ref_logits, label)

def hinge_loss(logit_scale, ref, candidates, label, device, margin=0.05):
    candidates = candidates.reshape(-1, 2, *candidates.shape[1:])
    ref_logits = torch.sum(ref * candidates[:,0], dim=1) - torch.sum(ref * candidates[:,1], dim=1)
    acc = calc_acc(ref_logits, label)
    y_rounded = torch.round(label) 
    y_transformed = -1 * (1 - 2 * y_rounded) 
    return torch.max(torch.zeros(ref_logits.shape).to(device), margin + (ref_logits * y_transformed)).sum(), acc

        
def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
        
def get_weight(args, losses, ema_acc):
    if args.task_weight == 'plain':
        return [1] * 4
    else:
        acc_list = [losses['2AFC_acc'], losses['Text-Images-AFC_acc'], losses['Text-2AFC_acc'], losses['IQA_acc']]
        ema_acc = [0.9 * ema_acc[idx] + 0.1 * acc_list[idx] for idx in range(4)]
        weight_list = [1/ema_acc[idx] for idx in range(4)]
        return [weight/sum(weight_list) for weight in weight_list]

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tokenizer, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    model.train()
    if args.distill:
        dist_model.eval()
    finished_loaders, iter_list = list(), list()
    for idx, dataloader in enumerate(data['train'].dataloaders):
        # if data['train'].sampler_list[idx]:
        #     data['train'].sampler_list[idx].set_epoch(epoch)
        iter_list.append(iter(dataloader))
        finished_loaders.append(False)

    num_batches_per_epoch = data['train'].num_batches // (args.accum_freq * 4)
    sample_digits = math.ceil(math.log(data['train'].num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    ema_acc = [0.0] * 4
    # for type_id, dataloader in enumerate(data['train'].dataloaders):
    #    print(type_id, len(dataloader))
            
    #for idx, batches in enumerate(itertools.zip_longest(*iter_list)):
    # for idx, batches in enumerate(zip(*iter_list)):
    #     for type_id, batch in enumerate(batches):
    i  = -1    
    # while not all(finished_loaders):
    for i in range(len(data['train'].dataloaders[-1])):

        losses = dict() 
        losses['reg_loss'] = 0
        optimizer.zero_grad()
        # i += 1
        for type_id, dataloader in enumerate(data['train'].dataloaders):
            try:
                
                batch_type = data['train'].dataloader_types[type_id]
                batch = next(iter_list[type_id])
                if batch_type == '2AFC':
                    ref, img_1, img_2, label, _ = batch
                    with torch.no_grad():
                        if args.distill:
                            orig = unwrap_model(dist_model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                        
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    ref = unwrap_model(model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                    if args.distill:
                        losses['reg_loss'] += F.mse_loss(orig, ref)
                    
                    if args.loss == 'CL':
                        label = torch.arange(img_1.shape[0]) * 2 + label
                    
                elif batch_type == 'Text-Images-AFC':
                    img_1, img_2, label, ref = batch
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    with torch.no_grad():
                        if args.distill:
                            orig = unwrap_model(dist_model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                        ref = unwrap_model(model).encode_text(ref.to(device=device, non_blocking=True), normalize=True)# orig = unwrap_model(dist_model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                         
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    if args.distill:
                        losses['reg_loss'] += F.mse_loss(orig, candidates)
                    
                    if args.loss == 'CL':
                        label = torch.arange(img_1.shape[0]) * 2 + label
                elif batch_type == 'Text-2AFC':
                    ref, cap_1, cap_2, label = batch
                    candidates = torch.stack((cap_1, cap_2), dim=1).to(device=device, non_blocking=True)
                    with torch.no_grad():
                        candidates = unwrap_model(model).encode_text(candidates.view(-1, cap_1.shape[-1]), normalize=True)
                        if args.distill:
                            orig = unwrap_model(dist_model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                    
                    ref = unwrap_model(model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                    if args.distill:
                        losses['reg_loss'] += F.mse_loss(orig, ref)
                    
                    if args.loss == 'CL':
                        label = torch.arange(cap_1.shape[0]) * 2 + label
                        
                elif batch_type == 'IQA':
                    img_1, img_2, label, _ = batch
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    if args.clip_iqa:
                        prompt = ['A high-quality photo.', 'A low-quality photo.']
                        if random.randint(0,1) == 1:
                            prompt = tokenizer(prompt[1])
                            label = 1 - label
                        else:
                            prompt = tokenizer(prompt[0])
        
                    else:
                        prompt = tokenizer(['A high-quality photo.'])
                        
                    with torch.no_grad():
                        prompt = prompt.unsqueeze(0).repeat(img_1.shape[0],1).to(args.device, non_blocking=True)
                        ref = unwrap_model(model).encode_text(prompt, normalize=True)
                        if args.distill:
                            orig = unwrap_model(dist_model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    if args.distill:
                        losses['reg_loss'] += F.mse_loss(orig, candidates)
                    
                    if args.loss == 'CL':
                        label = torch.arange(img_1.shape[0]) * 2 + label
                i_accum = i // args.accum_freq
                step = num_batches_per_epoch * epoch + i_accum

                if not args.skip_scheduler:
                    scheduler(step)

                data_time_m.update(time.time() - end)
                logit_scale = unwrap_model(model).logit_scale
                label = label.to(device, dtype=torch.long, non_blocking=True)
                if args.loss == 'CL':
                    losses[f'{batch_type}_loss'], losses[f'{batch_type}_acc'] = contrastive_loss(logit_scale, ref, candidates, label, device)
                elif args.loss == 'BL':
                    losses[f'{batch_type}_loss'], losses[f'{batch_type}_acc'] = binary_loss(logit_scale, ref, candidates, label, device)
                else:
                    losses[f'{batch_type}_loss'], losses[f'{batch_type}_acc'] = hinge_loss(logit_scale, ref, candidates, label, device)
                    
                data['train'].running_avg_loss[type_id] = 0.9 * data['train'].running_avg_loss[type_id] + \
                                                          0.1 * losses[f'{batch_type}_loss'].item()
                # losses[f'{batch_type}_normalized_loss'] = losses[f'{batch_type}_loss']/data['train'].running_avg_loss[type_id]

            except Exception as e:
                print('ERROR')
                print(e)
                import traceback
                traceback.print_exc() 
                # finished_loaders[type_id] = True
                
        try:        
            # losses['normalized_loss'], losses['loss'] = 0, 0  
            losses['loss'] = 0    
            weights = get_weight(args, losses, ema_acc) #[2/9, 4/9, 2/9, 1/9]
            for batch_idx, batch_type in enumerate(data['train'].dataloader_types):  
                # losses['normalized_loss'] += weights[batch_idx] * losses[f'{batch_type}_normalized_loss']
                losses['loss'] += weights[batch_idx] * losses[f'{batch_type}_loss']
             
            if args.normalize:
                loss_to_backward = losses['normalized_loss']
            else:
                loss_to_backward = losses['loss']          
            if args.distill:
                backward(loss_to_backward+ args.reg_wd * losses['reg_loss'], scaler)
            else:
                backward(loss_to_backward, scaler)           
            
            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                if (i + 1) % args.accum_freq == 0:
                    optimizer.step()         # Update model parameters
                    optimizer.zero_grad()    # Reset gradients for next accumulation

            # if args.accum_freq > 1:
                # accum_images, accum_texts, accum_features = [], [], {}

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            # with torch.no_grad():
            #     unwrap_model(model).logit_scale.clamp_(0, math.log(100))

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i_accum + 1
            if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch) and (i + 1) % args.accum_freq == 0:
                batch_size = img_1.shape[0]
                num_samples = batch_count * batch_size * 4 * args.accum_freq * args.world_size
                samples_per_epoch = data['train'].num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    if type(val) == torch.Tensor:
                        val = val.item()
                    losses_m[key].update(val, batch_size)

                # logit_scale_scalar = logit_scale.item()
                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )
                samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
                logging.info(
                    f"Task: {batch_type:>15} "
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
                    # f"Logit Scale: {logit_scale_scalar:.3f} " 
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "task_type": batch_type,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    # "scale": logit_scale_scalar,
                    "lr": optimizer.param_groups[0]["lr"]
                }            
                log_data.update({name:val.val for name,val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}
                
                if is_master(args) and i_accum % args.val_frequency == args.val_frequency - 1:
                    evaluate(model, data, epoch, args, tb_writer=tb_writer, tokenizer=tokenizer)
                    
                if tb_writer is not None:
                    for name, val in log_data.items():
                        tb_writer.add_scalar(name, val, step)
                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc() 
            # finished_loaders = [True]*4


def check_data(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tokenizer, tb_writer=None):
    """Mimics a training epoch but without any model."""

    # device = torch.device(args.device)
    # autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    print(args.train_data)

    if args.train_data == 'synthetic':

        data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = data['train'].dataloader
        iter_ = iter(dataloader)
        
        startt = time.time()

        for i in range(len(dataloader)):
            
            oldt = time.time()
            batch = next(iter_)
            currt = time.time() - oldt
            tott = time.time() - startt
            if is_master(args):
                print(f'it={i} [time] iter={currt:.3f} s tot={tott:.3f}')
            
    else:
        model.train()
        if args.distill:
            dist_model.eval()
        finished_loaders, iter_list, data['train'].running_avg_loss = list(), list(), list()
        for idx, dataloader in enumerate(data['train'].dataloaders):
            if data['train'].sampler_list[idx]:
                data['train'].sampler_list[idx].set_epoch(epoch)
            # iter_list.append(iter(dataloader))
            data['train'].running_avg_loss.append(0)
            finished_loaders.append(False)

        # #data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
        # #dataloader = data['train'].dataloader
        # num_batches_per_epoch = data['train'].num_batches // args.accum_freq
        # sample_digits = math.ceil(math.log(data['train'].num_samples + 1, 10))

        # if args.accum_freq > 1:
        #     accum_images, accum_texts, accum_features = [], [], {}

        # losses_m = {}
        # batch_time_m = AverageMeter()
        # data_time_m = AverageMeter()
        # end = time.time()

        if is_master(args):
            print(f'batch size={args.batch_size}')

        for type_id, dataloader in enumerate(data['train'].dataloaders):
            print(type_id, len(dataloader))

        i = -1
        startt = time.time()
        while not all(finished_loaders):
            
            for type_id, dataloader in enumerate(data['train'].dataloaders):
                
                try:
                    batch_type = data['train'].dataloader_types[type_id]
                    oldt = time.time()
                    batch = next(iter_list[type_id])
                    currt = time.time() - oldt
                    i+=1

                    tott = time.time() - startt
                    if is_master(args):
                        print(
                            f'it={i} type={type_id}'  # ds={dataloader.dataset.datasets[0].name}
                            f' [time] iter={currt:.3f} s tot={tott:.3f}'
                            )
                    
                except Exception as e:
                    print(e)
                    finished_loaders[type_id] = True
                    return


        
def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    metrics.update({})

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
    
    samples_per_val = data['val'].num_samples
    cumulative_accuracy, cumulative_num_samples = 0, 0
    cumulative_loss = 0.0
    i = -1
    with torch.inference_mode():
        for idx, dataloader in enumerate(data['val'].dataloaders):
            # data['val'].sampler_list[idx].set_epoch(epoch)
            batch_type = data['val'].dataloader_types[idx]
            correct_samples, num_samples = 0, 0
            print(len(dataloader))
            for batch in dataloader:
                i+=1
                if batch_type == '2AFC':
                    ref, img_1, img_2, label, _ = batch
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    ref = unwrap_model(model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                    
                elif batch_type == 'Text-Images-AFC':
                    img_1, img_2, label, ref = batch
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    ref = unwrap_model(model).encode_text(ref.to(device=device, non_blocking=True), normalize=True)
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)
                    
                elif batch_type == 'Text-2AFC':
                    ref, cap_1, cap_2, label = batch
                    candidates = torch.stack((cap_1, cap_2), dim=1).to(device=device, non_blocking=True)
                    candidates = unwrap_model(model).encode_text(candidates.view(-1, cap_1.shape[-1]), normalize=True)
                    ref = unwrap_model(model).encode_image(ref.to(device=device, dtype=input_dtype, non_blocking=True), normalize=True)
                        
                elif batch_type == 'IQA':
                    img_1, img_2, label, _ = batch
                    candidates = torch.stack((img_1, img_2), dim=1).to(device=device, dtype=input_dtype, non_blocking=True)
                    prompt = tokenizer(['A high-quality photo.']).unsqueeze(0).repeat(img_1.shape[0],1).to(args.device, non_blocking=True)
                    ref = unwrap_model(model).encode_text(prompt, normalize=True)
                    candidates = unwrap_model(model).encode_image(candidates.view(-1, *img_1.shape[1:]), normalize=True)

                with autocast():
                    total_loss =  F.cross_entropy(ref @ candidates.T, label.to(args.device, dtype=torch.long, non_blocking=True))
                    
                # for idx in range(ref_logits.shape[0]):
                #     pred = 0 if ref_logits[idx, idx*2] > ref_logits[idx, idx*2+1] else 1
                #     correct_samples += (pred == label[idx].item() - idx*2)
                #     cumulative_correct_samples += (pred == label[idx].item() - idx*2)
                candidates = candidates.reshape(-1, 2, *candidates.shape[1:])
  
                ref_logits = torch.sum(ref * candidates[:,0], dim=1) - torch.sum(ref * candidates[:,1], dim=1)
        
                accuracy = calc_acc(ref_logits, label).item()
                cumulative_accuracy += accuracy
                cumulative_loss += total_loss * ref.shape[0]
                num_samples = ref.shape[0]
                cumulative_num_samples += ref.shape[0]
                
            if is_master(args):
                logging.info(
                    f"accuracy:{round(accuracy, 4):>6} "
                    f"Task: {batch_type:>15} "
                    f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                    f"Clip Loss: {cumulative_loss / cumulative_num_samples:.6f}\t")

            loss = cumulative_loss / num_samples
            metrics.update({"accuracy": cumulative_accuracy/4, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics
    print(metrics)
    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            num_batches_per_epoch = data['train'].num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
