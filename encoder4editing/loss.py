from lib.loss_interface import Loss, LossInterface


class Encoder4EditingLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_softplus_loss(dict["d_adv"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(dict["id_source"], dict["id_recon"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L2_loss(dict["I_source"], dict["I_recon"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # lpips loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(dict["I_source"], dict["I_recon"])
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        L_real = Loss.get_softplus_loss(dict["d_real"], True)
        L_fake = Loss.get_softplus_loss(dict["d_fake"], False)
        L_reg = Loss.get_r1_reg(dict["d_real"], dict["w_real"])
        L_D = L_real + L_fake + L_reg
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        