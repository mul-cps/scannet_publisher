# https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
#
# Copyright (c) 2017
# Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber,
# Thomas Funkhouser, Matthias Niessner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import struct
import zlib

import cv2
import numpy as np

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort',
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == 'zlib_ushort':
            return zlib.decompress(self.depth_data)
        if compression_type == 'raw_ushort':
            return self.depth_data
        raise NotImplementedError(f'Unsupported depth compression: {compression_type}')

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type in ('jpeg', 'png'):
            return self._decompress_color_imdecode()
        raise NotImplementedError(f'Unsupported color compression: {compression_type}')

    def _decompress_color_imdecode(self) -> np.ndarray:
        buf = np.frombuffer(self.color_data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise NotImplementedError('Failed to decompress color image')
        return bgr


class SensorData:
    def __init__(self, filename: str):
        self.version = 4
        self.filename = filename
        self._file = open(filename, 'rb')
        self._load_header()

    def _load_header(self):
        f = self._file

        version = struct.unpack('I', f.read(4))[0]
        assert self.version == version

        strlen = struct.unpack('Q', f.read(8))[0]
        self.sensor_name = b''.join(
            struct.unpack('c' * strlen, f.read(strlen))
        ).decode('utf-8', errors='ignore')

        self.intrinsic_color = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_color = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.intrinsic_depth = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_depth = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)

        self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
        self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]

        self.color_width = struct.unpack('I', f.read(4))[0]
        self.color_height = struct.unpack('I', f.read(4))[0]
        self.depth_width = struct.unpack('I', f.read(4))[0]
        self.depth_height = struct.unpack('I', f.read(4))[0]
        self.depth_shift = struct.unpack('f', f.read(4))[0]

        self.num_frames = struct.unpack('Q', f.read(8))[0]

    def __iter__(self):
        return self._frame_generator()

    def _frame_generator(self):
        for _ in range(self.num_frames):
            frame = RGBDFrame()
            frame.load(self._file)
            yield frame

    def close(self):
        self._file.close()
