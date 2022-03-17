import torch
from lib import utils
from lib.model_interface import ModelInterface
from encoder4editing.loss import Encoder4EditingLoss
from encoder4editing.nets import Encoder4EditingGenerator
from submodel.discriminator import LatentCodesDiscriminator # with BCE


class Encoder4Editing(ModelInterface):
    def initialize_models(self):
        self.G = Encoder4EditingGenerator(arcface_path=self.args.arcface_path, batch_per_gpu=self.args.batch_per_gpu, stylegan_path=self.args.stylegan_path, stylegan_size=self.args.stylegan_size).cuda(self.gpu).train()
        self.D = LatentCodesDiscriminator().cuda(self.gpu).train()

    def set_loss_collector(self):
        self._loss_collector = Encoder4EditingLoss(self.args)

    def train_step(self, global_step):
        # load batch
        I_source = self.load_next_batch()

        self.dict = {
            "I_source": I_source,
        }

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        # run D
        self.run_D()

        # update D
        loss_D = self.loss_collector.get_loss_D(self.dict)
        utils.update_net(self.opt_D, loss_D)

        return [self.dict["I_source"], self.dict["I_recon"]]

    def run_G(self):
        I_recon, w_fake = self.G(self.dict["I_source"])
        d_adv = self.D(w_fake)
        id_source = self.G.get_id(self.dict["I_source"])
        id_recon = self.G.get_id(I_recon)

        self.dict["I_recon"] = I_recon
        self.dict["w_fake"] = w_fake
        self.dict["d_adv"] = d_adv
        self.dict["id_source"] = id_source
        self.dict["id_recon"] = id_recon

    def run_D(self):

        self.dict["w_real"] = self.G.get_w_from_random_z()
        self.dict["w_real"].requires_grad_()
        d_real = self.D(self.dict["w_real"])
        d_fake = self.D(self.dict["w_fake"].detach())

        self.dict["d_real"] = d_real # [8, 16, 1]
        self.dict["d_fake"] = d_fake

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source)[0]
        utils.save_image(self.args, step, "valid_imgs", [self.valid_source, Y])

    def save_image(self, result, step):
        utils.save_image(self.args, step, "imgs", result)

    @property
    def loss_collector(self):
        return self._loss_collector
        