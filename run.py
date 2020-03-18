import cv2
import DAIN_networks
import EDVR_networks.archs.EDVR_arch as EDVR_arch
import os
import numpy as np
import shutil
import torch
import EDVR_utils.util as EDVR_util
import EDVR_data.util as EDVR_data_util
from glob import glob
from imageio import imread, imsave
from torch.autograd import Variable

# DAIN settings
DAIN_model_name = 'best.pth'

# EDVR settings
EDVR_model_name = 'cinepak_small2.pth'
EDVR_N_in = 7

video_paths = []
for path in glob('input/*'):
    name = os.path.basename(path)
    print(name)
    if os.path.isfile(path):
        output_path = 'tmp/{:s}'.format(name)
        os.makedirs(output_path, exist_ok = True)
        video_paths.append(output_path)
        if not os.path.isfile('{:s}/00000001.png'.format(output_path)):
            os.system('ffmpeg -i {:s} {:s}/%08d.png -hide_banner'.format(path, output_path));

EDVR_model = EDVR_arch.EDVR(64, EDVR_N_in, 8, 5, 10, predeblur=False, HR_in=False)
EDVR_model.load_state_dict(torch.load('EDVR_models/{:s}'.format(EDVR_model_name)), strict=True)
EDVR_model.eval()
EDVR_model = EDVR_model.to(torch.device('cuda'))

DAIN_model = DAIN_networks.__dict__['DAIN'](channel=3,
                            filter_size=4,
                            timestep=0.5,
                            training=False)
DAIN_model = DAIN_model.cuda()
DAIN_dict = torch.load('DAIN_models/{:s}'.format(DAIN_model_name))

DAIN_model_dict = DAIN_model.state_dict()

# 1. filter out unnecessary keys
DAIN_dict = {k: v for k, v in DAIN_dict.items() if k in DAIN_model_dict}
# 2. overwrite entries in the existing state dict
DAIN_model_dict.update(DAIN_dict)
# 3. load the new state dict
DAIN_model.load_state_dict(DAIN_model_dict)
DAIN_model_dict = None

DAIN_model = DAIN_model.eval()

for path in video_paths:
    name = os.path.basename(path)
    length = len(glob(path + '/*.png'))
    sr_output_path = 'tmp/{:s}_sr_out/'.format(name)
    os.makedirs(sr_output_path, exist_ok = True)
    interp_output_path = 'tmp/{:s}_interp_out/'.format(name)
    os.makedirs(interp_output_path, exist_ok = True)

    if not os.path.isfile('{:s}/00000001.png'.format(sr_output_path)):
        frames = []
        for input_frame_number in range(1, length + 1):
            frame_path = '{:s}/{:08d}.png'.format(path, input_frame_number)
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=2)
            if frame.shape[2] > 3:
                frame = frame[:, :, :3]
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        frames = frames[:, :, :, [2, 1, 0]]
        frames = torch.from_numpy(np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))).float()

        for frame_idx, _ in enumerate(frames):
            select_idx = EDVR_data_util.index_generation(frame_idx, len(frames), EDVR_N_in, padding="replicate")
            input = frames.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(torch.device('cuda'))
            output = EDVR_util.single_forward(EDVR_model, input)
            output = EDVR_util.tensor2img(output.squeeze(0))
            output = cv2.resize(output, None, fx=0.625, fy=0.625, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('{:s}/{:08d}.png'.format(sr_output_path, frame_idx + 1), output)

    if not os.path.isfile('{:s}/00000001.png'.format(interp_output_path)):
        output_frame_number = 1
        for input_frame_number in range(1, length):
            frame_0_path = '{:s}/{:08d}.png'.format(sr_output_path, input_frame_number)
            frame_1_path = '{:s}/{:08d}.png'.format(sr_output_path, input_frame_number + 1)
            frame_0 =  torch.from_numpy( np.transpose(imread(frame_0_path) , (2,0,1)).astype("float32")/ 255.0).type(torch.cuda.FloatTensor)
            frame_1 =  torch.from_numpy( np.transpose(imread(frame_1_path) , (2,0,1)).astype("float32")/ 255.0).type(torch.cuda.FloatTensor)

            shutil.copyfile(frame_0_path, '{:s}/{:08d}.png'.format(interp_output_path, output_frame_number))
            output_frame_number += 1
            interp_count = 1 # currently only doubling frame rate is supported
            for i in range(0, interp_count):
                y_ = torch.FloatTensor()

                intWidth = frame_0.size(2)
                intHeight = frame_0.size(1)
                channel = frame_0.size(0)
                if not channel == 3:
                    continue

                if intWidth != ((intWidth >> 7) << 7):
                    intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                    intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                    intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
                else:
                    intWidth_pad = intWidth
                    intPaddingLeft = 32
                    intPaddingRight= 32

                if intHeight != ((intHeight >> 7) << 7):
                    intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                    intPaddingTop = int((intHeight_pad - intHeight) / 2)
                    intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
                else:
                    intHeight_pad = intHeight
                    intPaddingTop = 32
                    intPaddingBottom = 32

                pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                torch.set_grad_enabled(False)
                frame_0 = Variable(torch.unsqueeze(frame_0,0))
                frame_1 = Variable(torch.unsqueeze(frame_1,0))
                frame_0 = pader(frame_0)
                frame_1 = pader(frame_1)

                frame_0 = frame_0.cuda()
                frame_1 = frame_1.cuda()
                y_s,offset,filter = DAIN_model(torch.stack((frame_0, frame_1),dim = 0))
                y_ = y_s[0]

                frame_0 = frame_0.data.cpu().numpy()
                y_ = y_.data.cpu().numpy()
                offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
                frame_1 = frame_1.data.cpu().numpy()

                frame_0 = np.transpose(255.0 * frame_0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
                y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
                offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
                filter = [np.transpose(
                    filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                    (1, 2, 0)) for filter_i in filter]  if filter is not None else None
                # X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))            
                imsave('{:s}/{:08d}.png'.format(interp_output_path, output_frame_number), np.round(y_).astype(np.uint8))
                output_frame_number += 1
            if output_frame_number == length - 1:
                shutil.copyfile(frame_1_path, '{:s}/{:08d}.png'.format(interp_output_path, output_frame_number))
                output_frame_number += 1
    os.system('ffmpeg -f image2 -r 30 -i {:s}/%08d.png -crf 0 output/{:s} -hide_banner'.format(interp_output_path, name));
            