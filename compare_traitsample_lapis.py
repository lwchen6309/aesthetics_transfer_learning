import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from LAPIS_histogram_dataloader import load_data, load_data_testpair, load_testdata
import argparse
from compare_traitsample import compute_mean_emd
from utils.losses import EarthMoverDistance
earth_mover_distance = EarthMoverDistance()


if __name__ == '__main__':    
    from utils.argflags import parse_arguments
    parser = parse_arguments(False)
    parser.add_argument('--output_dir', type=str, default='score_dict')
    parser.add_argument('--trait1', type=str, default=None)
    parser.add_argument('--value1', type=str, default=None)
    parser.add_argument('--trait2', type=str, default=None)
    parser.add_argument('--value2', type=str, default=None)
    args = parser.parse_args()
    print(args)
    
    if args.trait1 and args.trait2 is not None:
        args.trait, args.value = args.trait1, args.value1
        test_giaa_dataset, _ = load_testdata(args)
        args.trait, args.value = args.trait2, args.value2
        testc_giaa_dataset, _ = load_testdata(args)
    else:
        train_dataset, val_giaa_dataset, val_piaa_imgsort_dataset, test_giaa_dataset, test_piaa_imgsort_dataset, testc_giaa_dataset = load_data_testpair(args)
    
    mean_EMD = compute_mean_emd(test_giaa_dataset, testc_giaa_dataset, args)
    
    print(f'{args.trait} == {args.value}', mean_EMD)
    