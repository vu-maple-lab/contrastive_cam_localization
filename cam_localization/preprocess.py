from tqdm import tqdm
import numpy as np 
import json 
import argparse 
from pathlib import Path

# to be negative: tx, ty, tz, w, all should be > 10 
# to be positive; tx, ty, tz, w, all should be < 5

def preprocess(data_dir, validation=False):

    result = {}
    all_data = {}
    # first find the textfile we want & read the lines
    with open(str(data_dir), 'r') as f:
        lines = f.readlines()
    f.close()
    
    # create all_data, a dict where key: name of image, value: tx, ty, tz, fx, fy, fz, theta, vx, vy, vz 
    for line in lines:
        splitted = line.split()
        temp = [float(x) for x in (splitted[1:4] + splitted[7:])]
        all_data[splitted[0]] = np.array(temp)

    # loop 
    for key in tqdm(all_data):
        result[key] = {}
        result[key]['positive'] = list()
        result[key]['negative'] = list()

        # brute force search... sorry python 
        for _key, _val in all_data.items():

            # no need to check our current 
            if _key == key:
                continue 
            if not validation and not in_same_renders(_key, key):
                continue 

            # determine whether or not it's positive or negative 
            # positive_or_negative: True, False, None
            # trans_diff: a tuple (dtx, dty, dtz)
            # theta_diff: a float dtheta
            # add it to our result dictionary 
            positive_or_negative, trans_diff, theta_diff = check_positive_or_negative(_val, all_data[key])
            if positive_or_negative is None:
                continue
            if positive_or_negative is True:
                result[key]['positive'].append((_key,) + (trans_diff,) + (theta_diff,))
            elif positive_or_negative is False:
                result[key]['negative'].append((_key,) + (trans_diff,) + (theta_diff,))
    return result

def in_same_renders(query, original):
    query_type = query.split('_')[2]
    original_type = original.split('_')[2]
    return query_type == original_type

def check_positive_or_negative(query, original):
    query, original = query[:4], original[:4]
    trans_norm = np.linalg.norm(query[:3] - original[:3])
    deg_diff = np.abs(query[3] - original[3])
    if trans_norm > 10 or deg_diff > 10:
        bool_res = False 
    elif trans_norm < 5 and deg_diff < 10:
        bool_res = True 
    else:
        bool_res = None

    return bool_res, trans_norm.item(), deg_diff.item()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--threshold_val', type=float, default=0.05)
    args = parser.parse_args()

    threshold_val = args.threshold_val
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise Exception('data directory does not exist...')
    
    # first training 
    train_txtfile_path = data_dir / 'cleared_data' / 'cam_traj.txt'
    test_txtfile_path = data_dir / 'validation_data_dilated1' / 'Patient1_dilated.txt'
    print('Creating training dataset...')
    data = preprocess(train_txtfile_path)
    with open(str(data_dir / 'cleared_data' / 'clustered_cam_poses.json'), 'w') as f:
        json.dump(data, f)
    f.close()
    print('Creating testing dataset...')
    data = preprocess(test_txtfile_path, validation=True)
    with open(str(data_dir / 'validation_data_dilated1' / 'clustered_cam_poses.json'), 'w') as f:
        json.dump(data, f)
    f.close() 
    print("FINISHED!")
    