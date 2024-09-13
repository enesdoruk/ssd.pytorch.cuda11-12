import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import torch
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data import *
from data.config import *
from models.ssd import build_ssd
from utils.visualization import predict_visualization

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Inference With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC',
                    choices=['VOC', 'CS'],
                    type=str)
parser.add_argument('--pretrained_dir', default='weights/ssd.pth',
                    help='Directory for saving checkpoint models')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--size', default=300, type=int,
                    help='Image size for training')
parser.add_argument('--image', default=None,
                    help='image')
parser.add_argument('--video', default=None,
                    help='video')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='prediction_thresh')


cuda = torch.cuda.is_available()



def inference(image, transform, model):
    img, _, _ = transform(image)
    img = img[:, :, (2, 1, 0)]
    img = torch.from_numpy(img).permute(2, 0, 1)
    x = Variable(img.unsqueeze(0))
    if cuda: 
        x = x.cuda()

    h, w = img.shape[-1], img.shape[-2]
    with torch.no_grad():
        detections = model(x, 'test').data
        pred = []
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue

            scores = dets[:, 0].cpu().numpy()
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
        
            result = [boxes[i] for i in range(len(scores)) if scores[i] >= args.thresh]
            if len(result) != 0:
                pred.append(result)

    pred = torch.cat([p[0].unsqueeze(0) for p in pred], dim=0)
    res = predict_visualization(img, pred)
    
    return res


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    args = parser.parse_args()

    if args.dataset == 'VOC':
            cfg = voc
    elif args.dataset == 'CS':
        cfg = cs

    model = build_ssd(cfg['min_dim'], cfg['num_classes'])
    model_weights = torch.load(args.pretrained_dir)
    model.load_state_dict(model_weights, strict=True)

    transform = BaseTransform(args.size, MEANS)

    model = model.eval()
    if cuda:
        model = model.cuda()

    cudnn.benchmarks = True

    if args.image is not None:
        image = cv2.imread(args.image)
        
        start = time.time()
        out = inference(image, transform, model)
        cv2.imwrite('out.jpg', out)
        print(f"inference time: {time.time()-start}")
    
    if args.video is not None:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            print("Error establishing connection")

        frame_width = int(cfg['min_dim'])
        frame_height = int(cfg['min_dim'])
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        output_video_path = 'out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while capture.isOpened():
            ret, frame = capture.read()
            
            if not ret:
                break

            start = time.time()
            out = inference(frame, transform, model)
            print(f"Inference time: {time.time() - start:.2f} seconds")
    
            output_video.write(out)

        capture.release()
        output_video.release()
        print(f"Output video saved at {output_video_path}")
