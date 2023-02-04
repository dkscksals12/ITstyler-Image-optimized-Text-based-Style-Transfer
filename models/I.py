import torch
import clip
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys, os
from einops import repeat
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from f import adain, calc_mean_std

DECODER = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), # decoder_pho starts layer
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

VGG = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)), 
    nn.ReLU(),  # relu5-1  # 结束
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIP(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # for n, k in zip([input_dim] + h, h + [output_dim]):
        #     print(n, "\t" , k)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x.float())) if i < self.num_layers - 1 else layer(x.float())
        return x

class mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(*[nn.Linear(512, 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(),
                                        nn.Linear(512, 1024), nn.ReLU(),
                                        nn.Linear(1024, 4096), nn.ReLU(),
                                        nn.Linear(4096, 512*8*8), nn.ReLU()])
    def forward(self, x):
        return self.layers(x.float())


class Discriminator(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.enc = nn.Sequential(*list(VGG.children())[:31])
        self.cnn = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(), 
                                     nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3), nn.ReLU(), 
                                     nn.AdaptiveAvgPool2d(1)])
        self.fc = nn.Sequential(*[nn.Linear(1024, 1024), nn.ReLU(), 
                                    nn.Linear(1024, 1), nn.Sigmoid()])
    def get_patch(self, img):
        ret = []
        for b in range(img.shape[0]):
            for _ in range(4):
                x, y = np.random.randint(0, img.shape[2]-32), np.random.randint(0, img.shape[3]-32)
                ret.append(img[b, :, x:x+32, y:y+32].unsqueeze(0))
        ret = torch.cat(ret, dim=0)
        return ret

    def forward(self, patch, ins):
        f = self.enc(patch)
        f = self.cnn(f).squeeze()
        out = self.fc(torch.cat([f, ins], dim=1))
        
        return out

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

# model = MultiDiscriminator()
# print(model)

class Net(nn.Module):
    def __init__(self, encoder, decoder, MLP, CLIP, args) -> None:
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.clip = CLIP
        self.mse_loss = nn.MSELoss()
        self.decoder = decoder
        self.MappingNet = MLP
        self.end_layer = 4
        self.bce_loss = nn.BCELoss()
        # self.D = Discriminator
        self.tt = nn.Conv2d(1, 512, 1, 1, 0)

        self.upsample = torch.nn.Upsample(scale_factor=7, mode='bicubic', align_corners=True)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=8) # 关注背景
        self.avg_pool = torch.nn.MaxPool2d(kernel_size=8)
        

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      #vassert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def calc_style_loss1(self, gammar, beta, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        # input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(beta, target_mean) + \
               self.mse_loss(gammar, target_std)

    def clip_normalize(self, image):
        # image = F.interpolate(image,size=224,mode='bicubic')
        image = self.avg_pool(self.upsample(image))
        
        b, *_ = image.shape
        mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
        mean = repeat(mean.view(1, -1, 1, 1), '1 ... -> b ...', b=b) 
        std = repeat(std.view(1, -1, 1, 1), '1 ... -> b ...', b=b) 

        image = (image - mean) / std
        return image
    def forward(self, content, style, Istrain=True, ins=None):
        if Istrain:
            content_feats = self.encode_with_intermediate(content)
            style_feats = self.encode_with_intermediate(style)
            s_com = self.clip_normalize(style)
            s_clip = self.clip.clip_model.encode_image(self.clip_normalize(style))
        else:
            s_clip = torch.from_numpy(self.clip.clip_model.encode_text(clip.tokenize([style], truncate=True).cuda()).float().data.cpu().numpy()).cuda()
        s_clip1 = self.MappingNet(s_clip)

        S_c = self.encode(content)

        gammar, beta = s_clip1.contiguous().unsqueeze(2).unsqueeze(3).chunk(2, 1)

        # f = S_c.shape[-2] // s_clip1.shape[-2]
        # s_clip1 = F.interpolate(s_clip1, scale_factor=f, mode='bicubic', align_corners=True)

        t = adain(S_c, gammar, beta)
        gt = self.decoder(t)
        
        if not Istrain:
            return gt

        g_t_feats = self.encode_with_intermediate(gt)

        loss_s_a = self.calc_style_loss1(gammar, beta, style_feats[-1])

        loss_c = self.calc_content_loss(g_t_feats[-1], t)

        loss_s = 0
        for i in range(0, self.end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        return gt, loss_s_a, loss_c, loss_s







