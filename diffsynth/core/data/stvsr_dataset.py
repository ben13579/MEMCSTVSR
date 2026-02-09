from .unified_dataset import UnifiedDataset
from .operators import *


class STVSRDataset(UnifiedDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        space_scale=4, time_scale=8,
    ):
        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=tuple(data_file_keys),
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
            max_data_items=max_data_items,
        )
        self.space_scale = space_scale
        self.time_scale = time_scale

    def load_clip_operators(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        default_k=9,
        strict=True,
    ):
        frame_proc = ImageCropAndResize(
            height, width, max_pixels,
            height_division_factor, width_division_factor
        )

        return RouteByType(operator_map=[
            # ✅ 你的新 metadata 入口：dict -> LoadClip
            (dict, LoadClip(
                base_path=base_path,
                default_k=default_k,
                frame_processor=frame_proc,
                strict=strict,
            )),

            # ✅ 如果你直接丟 list[str]（例如已經 json.loads）
            (list, SequencialProcess(
                ToAbsolutePath(base_path) >> LoadImage() >> frame_proc
            )),

            # ✅ 保留原本：丟進來是單一路徑 str
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"),
                LoadImage() >> frame_proc >> ToList()),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                # 這裡用 default_k，start=0（給你「直接讀頭 k 張」的行為）
                LoadVideoClip(k=default_k, start=0, frame_processor=frame_proc, strict=strict)),
            ])),
        ])

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if key in self.special_operator_map:
                    data[key] = self.special_operator_map[key](data[key])
                elif key in self.data_file_keys:
                    data['GT'] = self.main_data_operator(data[key])
                    data['LQ'] = DownsampleVideo(
                        space_scale=self.space_scale,
                        time_scale=self.time_scale,
                    )(data['GT']) # LQ_i-2, LQ_i-1, LQ_i, LQ_i+1
                    data['GT'] = data['GT'][1:-1] # HQ_i-1, HQ_i, + 中間內插
        return data
