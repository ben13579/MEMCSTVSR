import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadVideoClip(DataProcessingOperator):
    def __init__(self, k=9, start=0, frame_processor=lambda x: x, strict=True):
        self.k = int(k)
        self.start = int(start)
        self.frame_processor = frame_processor
        self.strict = strict

    def __call__(self, video_path: str):
        reader = imageio.get_reader(video_path)

        try:
            n = int(reader.count_frames())
        except Exception:
            n = None

        # 嚴格模式：不足 k 直接炸，方便你早期抓資料/metadata 的 bug
        if self.strict and (n is not None) and (self.start + self.k > n):
            reader.close()
            raise ValueError(
                f"[LoadVideoClip] not enough frames: path={video_path}, "
                f"count={n}, start={self.start}, k={self.k}"
            )

        frames = []
        # 非嚴格：能讀多少讀多少（通常不建議 training 用）
        max_len = self.k if (n is None or self.strict) else max(0, min(self.k, n - self.start))

        for i in range(max_len):
            fid = self.start + i
            try:
                frame = reader.get_data(fid)
            except Exception:
                if self.strict:
                    reader.close()
                    raise
                else:
                    break
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)

        reader.close()
        return frames


class LoadClip(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        default_k=9,
        frame_processor=lambda x: x,
        strict=True,
        video_key="video",
        frames_key="frames",
        start_key="start",
        k_key="k",
    ):
        self.base_path = base_path
        self.default_k = int(default_k)
        self.frame_processor = frame_processor
        self.strict = strict
        self.video_key = video_key
        self.frames_key = frames_key
        self.start_key = start_key
        self.k_key = k_key

    def __call__(self, meta: dict):
        # Case 1: frames list（通常已經是 clip 長度=k 的 list[str]）
        if self.frames_key in meta and meta[self.frames_key] is not None:
            frames = meta[self.frames_key]

            # 如果你 CSV 讀進來還是字串（JSON list），這裡順便支援一下
            if isinstance(frames, str):
                import json
                frames = json.loads(frames)

            if not isinstance(frames, list):
                raise ValueError(f"[LoadClip] frames must be list/JSON-str, got {type(frames)}")

            abs_frames = [ToAbsolutePath(self.base_path)(p) for p in frames]
            return SequencialProcess(
                LoadImage() >> self.frame_processor
            )(abs_frames)

        # Case 2: video + start + k
        if self.video_key in meta and meta[self.video_key] is not None:
            video = ToAbsolutePath(self.base_path)(meta[self.video_key])
            start = int(meta.get(self.start_key, 0))
            k = int(meta.get(self.k_key, self.default_k))
            return LoadVideoClip(k=k, start=start, frame_processor=self.frame_processor, strict=self.strict)(video)

        raise ValueError(f"[LoadClip] meta must contain '{self.frames_key}' or '{self.video_key}'. meta={meta}")


class DownsampleVideo(DataProcessingOperator):
    def __init__(self, space_scale=4, time_scale=8, downsample_indexes=(0, 1, -2, -1)):
        self.space_scale = int(space_scale)
        self.time_scale = int(time_scale)
        self.downsample_indexes = list(downsample_indexes)

    def __call__(self, frames):
        # frames: list[PIL.Image]
        n = len(frames)
        if n == 0:
            return []

        # 把 index 正規化到 [0, n-1]，並過濾越界
        idxs = []
        for i in self.downsample_indexes:
            j = i if i >= 0 else n + i
            if 0 <= j < n:
                idxs.append(j)

        # 可選：去重但保持順序（避免 1 張 frame 時 0/ -1 都指到同一張）
        seen = set()
        idxs = [x for x in idxs if not (x in seen or seen.add(x))]

        downsampled_frames = []
        for j in idxs:
            frame = frames[j]
            w, h = frame.size
            new_w = max(1, w // self.space_scale)
            new_h = max(1, h // self.space_scale)
            downsampled_frames.append(
                frame.resize((new_w, new_h), resample=Image.BICUBIC)
            )
        return downsampled_frames
