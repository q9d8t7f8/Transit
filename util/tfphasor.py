
from cv2 import meanShift
import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
from .helper import definePsf, resamplingOperator, \
filterLaplacian,waveconvparam

class tfphasor(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5):
        super(tfphasor, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)
        
    def forward(self, feture_bxdxtxhxw):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        # for tbe, ten in zip(tbes, tens):
        #     assert tbe >= 0
        #     assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, 0, hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, 0, hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum//2)
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T



        # phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding='same')
        # data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        
        # datapad_2BDx2Tx2Hx2W = torch.zeros((2 * bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_2Dx2Tx2Hx2W = self.datapad_2Dx2Tx2Hx2W
        # create new variable
        datapad_B2Dx2Tx2Hx2W = datapad_2Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        # actually, because it is all zero so it is ok
        datapad_2BDx2Tx2Hx2W = datapad_B2Dx2Tx2Hx2W
        
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_2BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2
        
        ###########################################################3
        # Step 3: convolve with backprojection kernel
        
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        # datafre = torch.rfft(datapad_2BDx2Tx2Hx2W, 3, onesided=False)
        datafre = torch.fft.fftn(datapad_2BDx2Tx2Hx2W)
        datafre_real = datafre.real
        datafre_imag = datafre.imag
        
        re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        refre = torch.stack([re_real, re_imag], dim=4)

        refre = torch.stack([re_real, re_imag], dim=4)
        refre = torch.view_as_complex(refre)
        re = torch.fft.ifftn(refre)

        volumn_2BDxTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid]
        
        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :].real
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :].imag
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :].real
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :].imag
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW

def noise(data):
        gau = 0.05 + 0.03 * torch.randn_like(data) + data
        poi = 0.03 * torch.randn_like(data) * gau + gau
        return poi

if __name__ == '__main__':
    import cv2
    from einops import rearrange
    test_phasor = True
    path = '/storage/data/yuyh/chen_task/meas/person15-00000.hdr'
    if test_phasor:
        model = tfphasor(spatial= 32, crop=64, bin_len=0.01*8, \
                       wall_size=2)
        data = cv2.imread(path, -1)
        data = data / np.max(data)
        meas = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        meas = meas / np.max(meas)
        meas = rearrange(meas, '(t h) w ->t h w', t=600)
        meas = meas[:512, :, :]
        meas = torch.from_numpy(meas)
        print(torch.max(meas))
        meas = noise(meas)
        print(torch.max(meas))
        # meas = sio.loadmat(path)['meas']
        # meas = meas.astype(np.float32)
        K = 3
        for i in range(K):
             meas = (meas[::2, :, :] + meas[1::2, :, :]) / 2
             meas = (meas[:, ::2, :] + meas[:, 1::2, :]) / 2
             meas = (meas[:, :, ::2] + meas[:, :, 1::2]) / 2
        meas = rearrange(meas, 't h w ->1 1 t h w')
        # x = torch.from_numpy(meas)
        # x = x.astype(torch.float)
        model.todev(dev='cpu',dnum=1)
        print(torch.max(meas))
        x = noise(meas)
        print(torch.max(x))
        out = model(meas)
        re = out.detach().numpy()[0,0]
        p = np.max(re, axis=0)
        p = p/np.max(p)
        cv2.imwrite(f'{K}_downpf.png', p*255)
        print('done')
