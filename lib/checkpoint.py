import torch
import os

        
def load_checkpoint(isMaster, ckpt_path, model, optimizer):
    try:
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt_dict['model'], strict=False)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        step = ckpt_dict['step']
        
        if isMaster:
            print(f"Pretrained parameters are succesively loaded from {os.path.split(ckpt_path)[1]}")

        return step
    except:
        if isMaster:
            print(f"Failed to load checkpoint; start training from scratch")
        return 0


def save_checkpoint(args, model, optimizer, name, global_step):
    ckpt_dict = {}
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()
    ckpt_dict['step'] = global_step

    dir_path = f'./train_result/{args.run_id}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = f'{dir_path}/{name}_{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = f'{dir_path}/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        