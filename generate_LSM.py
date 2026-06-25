import math
import torch 
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import data_prepare as dp
from model import LSM_cnn


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Gaussian Processes on MNIST")
    parser.add_argument( "--feature_path", default='origin_data/feature/', type=str)
    parser.add_argument( "--label_path", default='origin_data/label/label1.tif', type=str)
    parser.add_argument( "--window_size", default=15, type=int)
    parser.add_argument( "--lr", default=0.0001, type=float)
    parser.add_argument( "--batch_size", default=128, type=int)
    parser.add_argument( "--epochs", default=300, type=int)
    parser.add_argument( "--slide_window", default=512, type=int)
    parser.add_argument( "--model_path", default='Result/best.pth', type=str)
    parser.add_argument( "--output_path", default='Result/lsm_test.tif', type=str)
    parser.add_argument( "--num_workers", default=2, type=int)
    args = parser.parse_args()
    return args


def predict_block(model, feature_block, window_size, batch_size, device):
    """
    Predict one padded feature block and return the unpadded probability block.
    """
    _, padded_h, padded_w = feature_block.shape
    block_h = padded_h - window_size + 1
    block_w = padded_w - window_size + 1
    window_view = np.lib.stride_tricks.sliding_window_view(
        feature_block, (window_size, window_size), axis=(1, 2)
    )
    result = np.empty(block_h * block_w, dtype=np.float32)

    for start in tqdm(range(0, result.size, batch_size), leave=False):
        end = min(start + batch_size, result.size)
        indices = np.arange(start, end)
        rows = indices // block_w
        cols = indices % block_w
        batch = window_view[:, rows, cols, :, :].transpose(1, 0, 2, 3).copy()
        images = torch.from_numpy(batch).float().to(device)
        result[start:end] = torch.softmax(model(images), dim=1)[:, 1].cpu().numpy()
    return result.reshape(block_h, block_w)


def iter_prefetched_blocks(tif_paths, window_size, block_size, num_workers):
    """
    Read feature blocks in parallel while inference consumes completed blocks.
    """
    if num_workers <= 1:
        yield from dp.iter_feature_blocks(tif_paths, window_size, block_size)
        return

    feature_paths = dp.get_feature_paths(tif_paths)
    width, height = dp.get_raster_size(feature_paths[0])
    stats = dp.get_feature_stats(feature_paths, block_size)
    windows = dp.feature_block_windows(width, height, block_size)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for window in windows:
            future = executor.submit(dp.read_feature_block, feature_paths, stats, window_size, *window)
            futures[future] = window
            if len(futures) >= num_workers:
                done = next(as_completed(futures))
                futures.pop(done)
                yield done.result()

        for done in as_completed(futures):
            yield done.result()


def main():
    args = parse_args()
    print('*******************************************generate LSM*******************************************')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_paths = dp.get_feature_paths(args.feature_path)
    n_feature = len(feature_paths)
    model = LSM_cnn(n_feature)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    output = dp.create_tif_like(args.label_path, args.output_path)
    with torch.no_grad():
        width, height = dp.get_raster_size(feature_paths[0])
        total_blocks = math.ceil(width / args.slide_window) * math.ceil(height / args.slide_window)
        blocks = iter_prefetched_blocks(args.feature_path, args.window_size, args.slide_window, args.num_workers)
        for x_off, y_off, block_w, block_h, feature_block in tqdm(blocks, total=total_blocks):
            prob_block = predict_block(model, feature_block, args.window_size, args.batch_size, device)
            dp.write_tif_block(output, x_off, y_off, prob_block[:block_h, :block_w])
    output.FlushCache()
    del output
    print('Finsih!')

if __name__=='__main__':
    main()
