
from munch import munchify
import torch
import time

import torch.multiprocessing as mp

from gui import gui_utils, slam_gui
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel

from utils.multiprocessing_utils import clone_obj
from utils.multiprocessing_utils import FakeQueue



class GaussianMapper(object):
    def __init__(self, config, args, slam, mapping_queue = None):
        self.config = config
        self.args = args
        self.slam = slam
        self.model_params = munchify(config["model_params"])
        self.opt_params = munchify(config["opt_params"])
        self.pipeline_params = munchify(config["pipeline_params"])

        self.gaussian_model = GaussianModel(0)

        self.use_spherical_harmonics = False
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        self.mapping_queue = mapping_queue
        self.use_gui = True
        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )

        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            print("GUI process started")
            time.sleep(5)

    def __call__(self, the_end = False):

        if not self.mapping_queue.empty():
            timestamp, image, depth, intrinsic, gt_pose = self.mapping_queue.get()
            self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                                gtcolor=image.squeeze(),
                                gtdepth=depth.detach().cpu().numpy()
                        )
            )

            # self.q_main2vis.put(
            #         gui_utils.GaussianPacket(
            #             gaussians=clone_obj(self.gaussians),
            #             current_frame=viewpoint,
            #             keyframes=keyframes,
            #             kf_window=current_window_dict,
            #         )
            # )