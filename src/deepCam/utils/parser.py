import argparse as ap

#dict helper for argparse
class StoreDictKeyPair(ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def parse_arguments():
    AP = ap.ArgumentParser()
    AP.add_argument("--wireup_method", type=str, default="nccl-openmpi", choices=["nccl-openmpi", "nccl-slurm", "nccl-slurm-pmi", "mpi"], help="Specify what is used for wiring up the ranks")
    AP.add_argument("--wandb_certdir", type=str, default="/opt/certs", help="Directory in which to find the certificate for wandb logging.")
    AP.add_argument("--run_tag", type=str, help="Unique run tag, to allow for better identification")
    AP.add_argument("--output_dir", type=str, help="Directory used for storing output. Needs to read/writeable from rank 0")
    AP.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to restart training from.")
    AP.add_argument("--data_dir_prefix", type=str, default='/', help="prefix to data dir")
    AP.add_argument("--max_inter_threads", type=int, default=1, help="Maximum number of concurrent readers")
    AP.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs to train")
    AP.add_argument("--save_frequency", type=int, default=100, help="Frequency with which the model is saved in number of steps")
    AP.add_argument("--validation_frequency", type=int, default=100, help="Frequency with which the model is validated")
    AP.add_argument("--max_validation_steps", type=int, default=None, help="Number of validation steps to perform. Helps when validation takes a long time. WARNING: setting this argument invalidates submission. It should only be used for exploration, the final submission needs to have it disabled.")
    AP.add_argument("--logging_frequency", type=int, default=100, help="Frequency with which the training progress is logged. If not positive, logging will be disabled")
    AP.add_argument("--training_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during training")
    AP.add_argument("--validation_visualization_frequency", type=int, default = 50, help="Frequency with which a random sample is visualized during validation")
    AP.add_argument("--local_batch_size", type=int, default=1, help="Number of samples per local minibatch")
    AP.add_argument("--channels", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], help="Channels used in input")
    AP.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "LAMB"], help="Optimizer to use (LAMB requires APEX support).")
    AP.add_argument("--start_lr", type=float, default=1e-3, help="Start LR")
    AP.add_argument("--adam_eps", type=float, default=1e-8, help="Adam Epsilon")
    AP.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    AP.add_argument("--loss_weight_pow", type=float, default=-0.125, help="Decay factor to adjust the weights")
    AP.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for linear LR warmup")
    AP.add_argument("--lr_warmup_factor", type=float, default=1., help="Multiplier for linear LR warmup")
    AP.add_argument("--lr_schedule", action=StoreDictKeyPair)
    AP.add_argument("--target_iou", type=float, default=0.82, help="Target IoU score.")
    AP.add_argument("--model_prefix", type=str, default="model", help="Prefix for the stored model")
    AP.add_argument("--enable_amp", action='store_true')
    AP.add_argument("--enable_jit", action='store_true')
    AP.add_argument("--enable_dali", action='store_true')
    AP.add_argument("--enable_nhwc", action='store_true')
    AP.add_argument("--data_augmentations", type=str, nargs='+', default=[], help="Data augmentations used. Supported are [roll, flip]")
    AP.add_argument("--enable_wandb", action='store_true')
    AP.add_argument("--resume_logging", action='store_true')
    AP.add_argument("--seed", default=333, type=int)
    pargs = AP.parse_args()
    
    return pargs
