import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, smoothloss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        # torch.save({
        #     'discriminator': self.discriminator.state_dict()
        # }, self.dis_weights_path)



class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__(config.MODEL, config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator(residual_blocks=8, init_weights=True)
        # discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            # discriminator = nn.DataParallel(discriminator , config.GPU)

        self.lam = config.LAMBDA

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()
        # adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        # self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        # self.add_module('perceptual_loss', perceptual_loss)
        # self.add_module('style_loss', style_loss)
        # self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        # self.dis_optimizer = optim.Adam(
        #     params=discriminator.parameters(),    
        #     lr=float(config.LR) * float(config.D2G_LR),
        #     betas=(config.BETA1, config.BETA2)
        # )

    def process(self, flows, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        # self.dis_optimizer.zero_grad()


        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():

        # process outputs
        outputs = self(flows, edges, masks)
        gen_loss = 0
        # dis_loss = 0


        # # discriminator loss
        # dis_input_real = flows
        # dis_input_fake = outputs.detach()
        # dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        # dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        # dis_real_loss = self.adversarial_loss(dis_real, True, True)
        # dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        # dis_loss = dis_loss + (dis_real_loss + dis_fake_loss) / 2

        # # generator adversarial loss
        # gen_input_fake = outputs
        # gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        # gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        # gen_loss = gen_loss + gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, flows) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        # gen_unmask_loss = self.l1_loss(outputs, flows) * self.config.UNMASK_LOSS_WEIGHT / torch.mean(masks)
        # gen_mask_loss = self.l1_loss(outputs*masks, flows*masks) * self.config.MASK_LOSS_WEIGHT / torch.mean(masks)
        gen_loss = gen_loss + gen_l1_loss

        # generator smooth loss
        if self.lam <= 0:
            gen_smo_loss = torch.tensor(0)
        else:
            gen_smo_loss = smoothloss(outputs, self.lam)* self.config.SMOOTH_LOSS_WEIGHT
        
        gen_loss = gen_loss + gen_smo_loss

        # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(outputs, flows)
        # gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        # gen_loss += gen_content_loss


        # generator style loss
        # gen_style_loss = self.style_loss(outputs * masks, flows * masks)
        # gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        # gen_loss += gen_style_loss


        # create logs
        logs = [
            # ("l_d2", dis_loss.item()),
            # ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_smo", gen_smo_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, 0, logs

    def forward(self, images, edge, mask):
        images_masked = (images * (1 - mask).float())
        inputs = torch.cat((images_masked, edge), dim=1)
        outputs = self.generator(images_masked, mask, edge)                                 # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        # dis_loss.backward()
        # self.dis_optimizer.step()

        # gen_loss = gen_loss.detach_().requires_grad_(True)
        gen_loss.backward()
        self.gen_optimizer.step()
