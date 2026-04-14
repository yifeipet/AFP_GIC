from typing import List, Dict, Tuple, Optional
import tempfile

import numpy as np
import torch

class HeaderHandler:
    def __init__(self):
        pass
    @staticmethod
    def check_img_size(img_size):
        assert len(img_size) == 2
        assert isinstance(img_size[0], int)
        assert isinstance(img_size[1], int)

    def encode(self, img_size: Tuple[int, int], y_hat: torch.Tensor, quality_ind: int) -> bytes:
        self.check_img_size(img_size)
        max_val = int(torch.max(torch.abs(y_hat)))
        info_list = [
            np.array(list(img_size), dtype=np.uint16),
            np.array(max_val, dtype=np.uint8),
            np.array(quality_ind, dtype=np.uint8),
        ]

        with tempfile.TemporaryFile() as f:
            for info in info_list:
                f.write(info.tobytes())
            f.seek(0)
            header_str = f.read()
        return header_str
    
    def byte_to_int(self, byte_string: bytes, dtype) -> int:
        return int(np.frombuffer(byte_string, dtype=dtype))
    
    def decode(self, header_byte_string: bytes) -> Dict:
        img_size = np.frombuffer(header_byte_string[:4], dtype=np.uint16)
        H, W = int(img_size[0]), int(img_size[1])
        max_sample = np.frombuffer(header_byte_string[4:5], dtype=np.uint8)
        max_sample = int(max_sample)
        quality_ind = np.frombuffer(header_byte_string[5:6], dtype=np.uint8)
        quality_ind = int(quality_ind)
        out_dict = {
            'img_size': (H, W),
            'max_sample': max_sample,
            'quality_ind': quality_ind,
        }
        return out_dict


def save_byte_strings(save_path: str, string_list: List) -> None:
    with open(save_path, 'wb') as f:
        for string in string_list:
            length = len(string)
            f.write(np.array(length, dtype=np.uint32).tobytes())
            f.write(string)

def load_byte_strings(load_path: str) -> List[bytes]:
    out_list = []
    with open(load_path, 'rb') as f:
        head = f.read(4)
        while head != b'':
            length = int(np.frombuffer(head, dtype=np.uint32)[0])
            out_list.append(f.read(length))
            head = f.read(4)
    return out_list


class IamgeInformation:
    def __init__(self, img_size: Tuple[int, int], max_sample: int, y_stride: int=16, z_stride: int=4) -> None:
        self.H, self.W = img_size
        self.max_sample = max_sample
        model_stride = y_stride * z_stride
        padH = int(np.ceil(self.H / model_stride) * model_stride)
        padW = int(np.ceil(self.W / model_stride) * model_stride)
        self.yH = padH // y_stride
        self.yW = padW // y_stride
        self.zH = padH // model_stride
        self.zW = padW // model_stride
