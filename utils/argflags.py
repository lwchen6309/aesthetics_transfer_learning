import argparse


def parse_arguments(parse=True):
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data splitting')
    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)  
    parser.add_argument('--trait_joint', action='store_false', dest='trait_disjoint', help='Disable disjoint trait')
    parser.add_argument('--use_cv', action='store_true', help='Enable cross-validation')

    parser.add_argument('--trainset', type=str, default='GIAA', choices=["GIAA", "sGIAA", "PIAA", "sGIAA-pair"])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--importance_sampling', action='store_true', help='Enable importance sampling for uniform score distribution')
    
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_on_piaa', action='store_true', help='Evaluation metric on PIAA')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')   
    
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_schedule_epochs', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)        
    
    if parse:
        return parser.parse_args()
    else:
        return parser


def parse_arguments_piaa(parse=True):
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data spliting')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--trait_joint', action='store_false', dest='trait_disjoint', help='Disable disjoint trait')

    parser.add_argument('--use_cv', action='store_true', help='Enable cross validation')
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=4)
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    # parser.add_argument('--pretrained_model', type=str, required=True)
    
    parser.add_argument('--is_eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--no_log', action='store_false', dest='is_log', help='Disable logging')
    
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr_schedule_epochs', type=int, default=5, help='Epochs after which to apply learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which to decay the learning rate')
    parser.add_argument('--max_patience_epochs', type=int, default=10, help='Max patience epochs for early stopping')    
    
    if parse:
        return parser.parse_args()
    else:
        return parser


def wandb_tags(args):
    tags = []
    if args.use_cv:
        tags += ["CV%d/%d"%(args.fold_id, args.n_fold)]
    if args.dropout is not None:
        tags += [f"dropout={args.dropout}"]
    if args.trait is not None and args.value is not None:
        tags = ["Trait specific", "Test trait: %s_%s"%(args.trait, args.value)]
    if not args.trait_disjoint:
        tags += ["Trait joint"]
    return tags
    

def model_dir(args):
    # Initialize the best test loss and the best model
    dirname = 'models_pth'
    if args.use_cv:
        dirname = os.path.join(dirname, 'random_cvs')
    if args.trait is not None and args.value is not None:
        if args.trait_disjoint:
            dirname = os.path.join(dirname, 'trait_disjoint_exp')
        else:
            dirname = os.path.join(dirname, 'trait_joint_exp')
    return dirname