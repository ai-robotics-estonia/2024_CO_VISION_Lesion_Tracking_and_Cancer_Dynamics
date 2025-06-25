import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#        U-NET Submodules
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.2):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('in', x.shape)
        # print('out', self.model(x).shape)
        return self.model(x)

# Added new class in order to fix the start of the bottleneck. 
class UNetMid_start(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetMid_start, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU()
        ]

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

# Updated bottleneck accordingly to the original paper. 
class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, 4, 1, 'same', bias=False)
        self.norm = nn.InstanceNorm3d(out_size)
        self.relu = nn.LeakyReLU(0.2)

        # self.model = nn.Sequential(*layers)


    def forward(self, x):

        y = self.conv(x)
        x = self.norm(y)
        x = self.relu(x)
        concatenated = torch.cat((x, y), dim=1)

        return concatenated
    
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('new')
        # print(x.shape)
        # print(skip_input.shape)
        x = self.model(x)
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)

        return x

class UNetDown2d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.2):
        super(UNetDown2d, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.2):
        super(UNetUp2d, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        
        x = torch.cat((x, skip_input), 1)

        return x
    
class UNetMid2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid2d, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, 4, 1, 'same', bias=False)
        self.norm = nn.InstanceNorm2d(out_size)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        y = self.conv(x)
        x = self.norm(y)
        x = self.relu(x)
        concatenated = torch.cat((x, y), dim=1)

        return concatenated


##############################
#        U-NET GLOBAL
##############################

class GeneratorUNetGlobal(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(GeneratorUNetGlobal, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.mid0 = UNetMid_start(256, 512)
        self.mid1 = UNetMid(512, 512)
        self.mid2 = UNetMid(1024, 512)
        self.mid3 = UNetMid(1024, 512)
        self.mid4 = UNetMid(1024, 256)
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)

        # Removed Tanh activation function in the final layer.
        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        m0 = self.mid0(d3)

        m1 = self.mid1(m0)
        m2 = self.mid2(m1)
        m3 = self.mid3(m2)
        m4 = self.mid4(m3)

        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        return self.final(u3), u3

##############################
#        U-NET LOCAL
##############################

class GeneratorLocalHat(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorLocalHat, self).__init__()

        #self.glob_pool = nn.Conv3d(128, 32, 1, 1, 0, bias=False)
    
        self.down1 = UNetDown2d(in_channels, 32)
        self.down2 = UNetDown2d(32, 64) 
        # <== Global Out level - 1ch
        self.down3 = UNetDown2d(64+out_channels, 128)  
        # <== Global Final feature level - 128ch
        self.mid1 = UNetMid2d(128, 128)
        self.mid2 = UNetMid2d(256, 256)
        self.mid3 = UNetMid2d(512, 128)
        self.mid4 = UNetMid2d(256, 64)
        # <== Global Final feature level 
        self.up1 = UNetUp2d(256, 64) 
        # <== Global Out level
        self.up2 = UNetUp2d(128+out_channels, 32)

        # Removed Tanh activation function in the final layer.
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, global_u3, global_out):
        # U-Net generator with skip connections from encoder to decoder

        # x - 2D image Tensor
        # global_u3 - 2D image Tensor
        
        d1 = self.down1(x)
        d2 = self.down2(d1)

        d2 = torch.cat((d2, global_out), 1)

        d3 = self.down3(d2)

        d3 += global_u3

        m1 = self.mid1(d3)
        m2 = self.mid2(m1)
        m3 = self.mid3(m2)
        m4 = self.mid4(m3)

        m4_u3 = torch.cat((m4, global_u3), 1)

        u1 = self.up1(m4_u3, d2)
        u2 = self.up2(u1, d1)
        
        return self.final(u2)

##############################
#        Discriminators
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding='same', bias=False)

    # def forward(self, img_A, img_B):
    #     # Concatenate image and condition image by channels to produce input
    #     img_input = torch.cat((img_A, img_B), 1)
    #     x = self.model(img_input)
    #     # x = F.pad(x, pad=(1,0,1,0,1,0))
    #     return self.final(x)

    def forward(self, img):
        x = self.model(img)
        final = self.final(x)
        return final

class Discriminator2d(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator2d, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
            *discriminator_block(1024, 1024),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv2d(1024, 1, 4, padding='same', bias=False)

    # def forward(self, img_A, img_B):
    #     # Concatenate image and condition image by channels to produce input
    #     img_input = torch.cat((img_A, img_B), 1)
    #     x = self.model(img_input)
    #     # x = F.pad(x, pad=(1,0,1,0,1,0))
    #     return self.final(x)
    def forward(self, img):
        return self.final(self.model(img))