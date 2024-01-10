import numpy as np
import logging
import cvcuda
import nvcv
import torch


class PreprocessorCvcuda:
    # docs_tag: begin_init_preprocessorcvcuda
    def __init__(self, scales, size, device_id, p=0.5, brightness=None, contrast=None, mode='train'):
        self.logger = logging.getLogger(__name__)
        # self.configer = configer
        self.scales=scales
        self.size=size
        # self.n_datasets = self.configer.get("n_datasets")
        self.p = p
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]

        self.device_id = device_id
        self.mean_tensor = torch.Tensor([0.3038, 0.3383, 0.3034])
        self.mean_tensor = self.mean_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.mean_tensor = cvcuda.as_tensor(self.mean_tensor, "NHWC")
        self.stddev_tensor = torch.Tensor([0.2071, 0.2088, 0.2090])
        self.stddev_tensor = self.stddev_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
        self.stddev_tensor = cvcuda.as_tensor(self.stddev_tensor, "NHWC")
        self.mode = mode

        self.logger.info("Using CVCUDA as preprocessor.")
        # docs_tag: end_init_preprocessorcvcuda

    # docs_tag: begin_call_preprocessorcvcuda
    def __call__(self, frame_nhwc_list, lb_nhwc_list=None):
        # docs_tag: begin_tensor_conversion
        # Need to check what type of input we have received:
        # 1) CVCUDA tensor --> Nothing needs to be done.
        # 2) Numpy Array --> Convert to torch tensor first and then CVCUDA tensor
        # 3) Torch Tensor --> Convert to CVCUDA tensor
         
        # if isinstance(frame_nhwc_list[0], torch.Tensor):
        #     frame_nhwc_list = [cvcuda.as_tensor(frame_nhwc, "NHWC") for frame_nhwc in frame_nhwc_list]
        # elif isinstance(frame_nhwc_list[0], np.ndarray):
        #     frame_nhwc_list = [cvcuda.as_tensor(
        #         torch.as_tensor(frame_nhwc).to(
        #             device="cuda:%d" % self.device_id, non_blocking=True
        #         ),
        #         "NHWC",
        #     ) for frame_nhwc in frame_nhwc_list]

        new_frame_nhwc_list = []
        for frame_nhwc in frame_nhwc_list:
            if isinstance(frame_nhwc, torch.Tensor):
                # print("!")
                new_frame_nhwc_list.append(cvcuda.as_tensor(frame_nhwc, "NHWC"))
            elif isinstance(frame_nhwc, np.ndarray):
                new_frame_nhwc_list.append(cvcuda.as_tensor(
                    torch.as_tensor(frame_nhwc).to(
                        device="cuda:%d" % self.device_id, non_blocking=True
                    ),
                    "NHWC",
                ))
            elif isinstance(frame_nhwc, list):
                for im in frame_nhwc:
                    if isinstance(im, torch.Tensor):
                        new_frame_nhwc_list.append(cvcuda.as_tensor(im.unsqueeze(0), "NHWC"))
                    elif isinstance(im, np.ndarray):

                        new_frame_nhwc_list.append(cvcuda.as_tensor(
                            torch.as_tensor(im).unsqueeze(0).to(
                                device="cuda:%d" % self.device_id, non_blocking=True
                            ),
                            "NHWC",
                        ))
                    else:
                        new_frame_nhwc_list.append(im)
            else:
                new_frame_nhwc_list.append(frame_nhwc)
                
                        
                        
        # docs_tag: end_tensor_conversion

        # docs_tag: begin_preproc_pipeline
        # Resize the tensor to a different size.
        # NOTE: This resize is done after the data has been converted to a NHWC Tensor format
        #       That means the height and width of the frames/images are already same, unlike
        #       a python list of HWC tensors.
        #       This resize is only going to help it downscale to a fixed size and not
        #       to help resize images with different sizes to a fixed size. If you have a folder
        #       full of images with all different sizes, it would be best to run this sample with
        #       batch size of 1. That way, this resize operation will be able to resize all the images.
        if self.mode == 'train':
            # if isinstance(lb_nhwc_list[0], torch.Tensor):
            #     lb_nhwc_list = [cvcuda.as_tensor(lb_nhwc, "NHWC") for lb_nhwc in lb_nhwc_list]
            # elif isinstance(lb_nhwc_list[0], np.ndarray):
            #     lb_nhwc_list = [cvcuda.as_tensor(
            #         torch.as_tensor(lb_nhwc).to(
            #             device="cuda:%d" % self.device_id, non_blocking=True
            #         ),
            #         "NHWC",
            #     ) for lb_nhwc in lb_nhwc_list]
            new_label_nhwc_list = []
            for lb_nhwc in lb_nhwc_list:
                if isinstance(lb_nhwc, torch.Tensor):
                    new_label_nhwc_list.append(cvcuda.as_tensor(lb_nhwc, "NHWC"))
                elif isinstance(lb_nhwc, np.ndarray):
                    new_label_nhwc_list.append(cvcuda.as_tensor(
                        torch.as_tensor(lb_nhwc).to(
                            device="cuda:%d" % self.device_id, non_blocking=True
                        ),
                        "NHWC",
                    ))
                elif isinstance(lb_nhwc, list):
                    for im in lb_nhwc:
                        if isinstance(im, torch.Tensor):
                            new_label_nhwc_list.append(cvcuda.as_tensor(im.unsqueeze(0), "NHWC"))
                        elif isinstance(im, np.ndarray):

                            new_label_nhwc_list.append(cvcuda.as_tensor(
                                torch.as_tensor(im).unsqueeze(0).to(
                                    device="cuda:%d" % self.device_id, non_blocking=True
                                ),
                                "NHWC",
                            ))
                        else:
                            new_label_nhwc_list.append(im)
                else:
                    new_label_nhwc_list.append(lb_nhwc)
                
            ims = []
            lbs = []
            for frame_nhwc, lb_nhwc in zip(new_frame_nhwc_list, new_label_nhwc_list):
                # print(type(frame_nhwc))
                # print(frame_nhwc.shape)
                n, h, w, _ = frame_nhwc.shape
                # print(frame_nhwc.shape)
                scale = np.random.uniform(min(self.scales), max(self.scales))
                if h*scale < self.size[0] or w*scale < self.size[1]:
                    h_rate = self.size[0] / h
                    w_rate = self.size[1] / w
                    scale = max([h_rate, w_rate])
                nh = min([int(h*scale+1), self.size[0]])
                nw = min([int(w*scale+1), self.size[1]])
                # im = cvcuda.resize(
                #     frame_nhwc,
                #     (
                #         frame_nhwc.shape[0],
                #         h*scale,
                #         w*scale,
                #         frame_nhwc.shape[3],
                #     ),
                #     cvcuda.Interp.LINEAR,
                # )
                im = cvcuda.resize(
                    frame_nhwc,
                    (
                        frame_nhwc.shape[0],
                        nh,
                        nw,
                        frame_nhwc.shape[3],
                    ),
                    cvcuda.Interp.LINEAR,
                )
                
                lb = cvcuda.resize(
                    lb_nhwc,
                    (
                        lb_nhwc.shape[0],
                        nh,
                        nw,
                        lb_nhwc.shape[3],
                    ),
                    cvcuda.Interp.NEAREST,
                )
                
                # crop_list = []
                # for _ in range(n):
                #     sh, sw = np.random.random(2)
                #     sh, sw = int(sh * (h*scale - self.size[0])), int(sw * (w*scale - self.size[1]))
                #     this_crop = torch.Tensor([sw, sh, self.size[1], self.size[0]]).reshape(1,1,1,4)
                #     crop_list.append(this_crop)
                sh, sw = np.random.random(2)
                sh, sw = int(sh * (nh - self.size[0])), int(sw * (nw - self.size[1]))
                # print(sh, sw)
                crop_tensor = nvcv.RectI(sw, sh, int(self.size[1]), int(self.size[0]))

                # crop_tensor = torch.cat(crop_list, dim=0).cuda(self.device_id)
                # crop_tensor = cvcuda.as_tensor(crop_tensor, "NHWC")
                # flip_tensor = torch.ones(im.shape[0], dtype=torch.int).cuda(self.device_id)
                # flip_tensor = cvcuda.as_tensor(flip_tensor, "N")

                # base_tensor = torch.Tensor([0,0,0])
                # base_tensor = base_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
                # base_tensor = cvcuda.as_tensor(base_tensor, "NHWC")
                # scale_tensor = torch.Tensor([1,1,1])
                # scale_tensor = scale_tensor.reshape(1, 1, 1, 3).cuda(self.device_id)
                # scale_tensor = cvcuda.as_tensor(scale_tensor, "NHWC")
                # im = cvcuda.crop_flip_normalize_reformat(
                #     im,
                #     (im.shape[0], self.size[1], self.size[0], im.shape[3]),
                #     np.uint8,
                #     "NHWC",
                #     crop_tensor,
                #     flip_tensor,
                #     base_tensor,
                #     scale_tensor,
                # )
                # lb = cvcuda.crop_flip_normalize_reformat(
                #     lb,
                #     out_shape = (lb.shape[0], self.size[1], self.size[0], lb.shape[3]),
                #     out_dtype = np.uint8,
                #     out_layout = "NHWC",
                #     rect = crop_tensor,
                #     flip_code = flip_tensor,
                #     base = base_tensor,
                #     scale = scale_tensor,
                # )
                im = cvcuda.customcrop(
                    im,
                    crop_tensor   
                )
                lb = cvcuda.customcrop(
                    lb,
                    crop_tensor
                )
                if np.random.random() > self.p:
                    im = cvcuda.flip(im, 1)
                    lb = cvcuda.flip(lb, 1)
                    
                    
                im = torch.as_tensor(
                    im.cuda(), device="cuda:%d" % self.device_id
                )
                lb = torch.as_tensor(
                    lb.cuda(), device="cuda:%d" % self.device_id
                )
                ims.append(im)
                lbs.append(lb)
            
            cropped_im=torch.cat(ims, dim=0)
            lbs=torch.cat(lbs, dim=0)
            ims = cvcuda.as_tensor(cropped_im, "NHWC")
            if self.brightness is not None and self.contrast is not None:
                brightness_rate = torch.Tensor([np.random.uniform(*self.brightness) for _ in range(ims.shape[0])]).cuda(self.device_id)
                contrast_rate = torch.Tensor([np.random.uniform(*self.contrast) for _ in range(ims.shape[0])]).cuda(self.device_id)
                brightness_rate = cvcuda.as_tensor(brightness_rate, "N")
                contrast_rate = cvcuda.as_tensor(contrast_rate, "N")
                contrastCenter = torch.Tensor([74 for _ in range(ims.shape[0])]).cuda(self.device_id)
                contrastCenter = cvcuda.as_tensor(contrastCenter, "N")
                ims = cvcuda.brightness_contrast(
                    ims,
                    brightness=brightness_rate,
                    contrast=contrast_rate,
                    contrast_center=contrastCenter
                )
                
            # Convert to floating point range 0-1.
            normalized = cvcuda.convertto(ims, np.float32, scale=1 / 255)

            # Normalize with mean and std-dev.
            normalized = cvcuda.normalize(
                normalized,
                base=self.mean_tensor,
                scale=self.stddev_tensor,
                flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
            )

            # Convert it to NCHW layout and return it.
            normalized = cvcuda.reformat(normalized, "NCHW")
            
        else:
            normalized = []
            lbs = lb_nhwc_list
            for ims in frame_nhwc_list:
                this_normalized = cvcuda.convertto(ims, np.float32, scale=1 / 255)

                # Normalize with mean and std-dev.
                this_normalized = cvcuda.normalize(
                    this_normalized,
                    base=self.mean_tensor,
                    scale=self.stddev_tensor,
                    flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
                )

                # Convert it to NCHW layout and return it.
                this_normalized = cvcuda.reformat(this_normalized, "NCHW")
                normalized.append(this_normalized)


        # Convert to floating point range 0-1

        # self.cvcuda_perf.pop_range()

        # Return 3 pieces of information:
        #   1. The original nhwc frame
        #   2. The resized frame
        #   3. The normalized frame.
        return normalized, lbs
        # docs_tag: end_preproc_pipeline