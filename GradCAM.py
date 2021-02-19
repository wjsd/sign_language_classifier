"""

Willis Sanchez-duPont
wsanchezdupont@g.hmc.edu

My implementation of Grad-CAM, copied from https://github.com/wjsd/Grad-CAM.git

"""

import numpy as np
import torch
import torch.nn as nn
import cv2

class GradCAM():
    """
    GradCAM

    Class that performs grad-CAM or guided grad-CAM on a given model (https://arxiv.org/abs/1610.02391).
    """
    def __init__(self,model,device='cuda',verbose=False):
        """
        Constructor.

        inputs:
            model - (torch.nn.Module) Indexable module to perform grad-cam on (e.g. torch.nn.Sequential)
            device - (str) device to perform computations on
            verbose - (bool) enables printouts for e.g. debugging
        """
        self.device = device
        self.model = model
        model.to(self.device) # push params to device
        self.model.eval()

        self.activations = torch.empty(0) # initialize activation maps
        self.grads = torch.empty(0) # initialize gradient maps
        self.results = torch.empty(0) # network output holder
        self.gradCAMs = torch.empty(0) # output maps
        self.fh = [] # module forward hook
        self.bh = [] # module backward hook

        self.guidehooks = [] # list of hooks on conv layers for guided backprop
        self.guidedGrads = torch.empty(0) # grads of class activations w.r.t. inputs
        self.deconv = None

        self.verbose = verbose

    def __call__(self,x,submodule=None,classes='all',guided=False,deconv=False):
        """
        __call__

        Compute class activation maps for a given input and submodule.

        inputs:
            x - (torch.Tensor) input image to compute CAMs for.
            submodule - (str or torch.nn.Module) name of module to get CAMs from OR submodule object to hook directly
            classes - (list) list of indices specifying which classes to compute grad-CAM for (MUST CONTAIN UNIQUE ELEMENTS). CAMs are returned in same order as specified in classes.
            guided - (None or True or list) Guided grad-CAM enabler. If None, no guided gradcam. If True,
                     apply guided backprop to all modules in self._modules.items() that have a
                     __class__.__name__ of ReLU. If list, apply hooks to all modules in the given list
            deconv - (bool) use 'deconvolution' backprop operation from https://arxiv.org/abs/1412.6806

        outputs:
            self.gradCAMs - (numpy.ndarray) numpy.ndarray of shape (batch,classes,u,v) where u and v are the activation map dimensions
        """
        x = x.to(self.device)
        x.retain_grad()

        if self.verbose:
            print('samples pushed to device')

        if submodule is None:
            self.activations = x # treat inputs as activation maps

        # register grad hook for vanilla grad-CAM
        self.set_gradhook(submodule)
        if self.verbose:
            print('CAM hooks set')

        # register hooks on ReLUs for guided backprop or 'deconvolution'
        self.deconv = deconv
        self.set_guidehooks(guided)

        # forward pass
        if guided and x.requires_grad==False:
            x.requires_grad = True

        self.results = self.model(x) # store result (already on device)
        self.gradCAMs = torch.empty(0)
        if self.verbose:
            print('model outputs computed')

        # grad and CAM computations
        summed = self.results.sum(dim=0) # sum out batch results. See note at top of file

        if classes == 'all':
            classes = [x for x in range(self.results.size(1))] # list of all classes
        for c in classes: # loop thru classes
            if c == classes[-1]:
                summed[c].backward()
            else:
                summed[c].backward(retain_graph=True) # retain graph to be able to backprop without calling forward again

            # see paper for details on math
            coeffs = self.grads.sum(-1).sum(-1) / (self.grads.size(2) * self.grads.size(3)) # coeffs has size = (batchsize, activations.shape(1)
            prods = coeffs.unsqueeze(-1).unsqueeze(-1)*self.activations # align dims to get appropriate linear combination of feature maps (which reside in dim=1 of self.activations)
            cam = torch.nn.ReLU()(prods.sum(dim=1)) # sum along activation dimensions (result size = batchsize x U x V)
            self.gradCAMs = torch.cat([self.gradCAMs, cam.unsqueeze(0).to('cpu')],dim=0) # add CAMs to function output variable
            self.model.zero_grad() # clear gradients for next backprop

            self.guidedGrads = torch.cat([self.guidedGrads, x.grad.cpu().unsqueeze(0)],dim=0) # save grad of class w.r.t. inputs
            x.grad.data = torch.zeros_like(x.grad.data) # clear out input grads as well

            if self.verbose:
                print('class {} gradCAMs computed'.format(c))

        # remove hooks
        self.bh.remove()
        if submodule is not None:
            self.fh.remove()

        if len(self.guidehooks) == 0:
            for hook in self.guidehooks:
                hook.remove()
        if self.verbose:
            print('all hooks removed')

        self.gradCAMs = self.gradCAMs.permute(1,0,2,3).detach().numpy() # batch first, requires_grad=False, and convert to numpy array
        self.guidedGrads = self.guidedGrads.permute(1,0,2,3,4).numpy() # batch,class,channel,height,width

        if self.verbose:
            print('self.gradCAMs.shape =',self.gradCAMs.shape)
            if guided:
                print('self.guidedGrads.shape =',self.guidedGrads.shape)

        # determine shape of new tensor
        if guided:
            resizedGradCAMs = np.zeros((self.gradCAMs.shape[0],self.gradCAMs.shape[1],3,x.shape[-2],x.shape[-1]))
        else:
            resizedGradCAMs = np.zeros((self.gradCAMs.shape[0],self.gradCAMs.shape[1],x.shape[-2],x.shape[-1]))

        # fill out resizedGradCAMs
        for i in range(self.gradCAMs.shape[0]): # batch
            for j in range(self.gradCAMs.shape[1]): # class
                resizedcam = cv2.resize(self.gradCAMs[i][j],(x.shape[-2],x.shape[-1]))
                if guided:
                    resizedGradCAMs[i][j] = resizedcam.reshape(1,*resizedcam.shape) * self.guidedGrads[i][j]
                else:
                    resizedGradCAMs[i][j] = resizedcam

                if self.verbose:
                    print('resizedGradCAMs.shape =',resizedGradCAMs.shape)
                    print('resizedGradCAMs.min() =',resizedGradCAMs.min())
                    print('resizedGradCAMs.max() =',resizedGradCAMs.max())

                resizedGradCAMs[i][j] = resizedGradCAMs[i][j] - np.min(resizedGradCAMs[i][j])
                resizedGradCAMs[i][j] = resizedGradCAMs[i][j] / np.max(resizedGradCAMs[i][j])

        self.gradCAMs = resizedGradCAMs
        if self.verbose:
            print('self.gradCAMs.shape =',self.gradCAMs.shape)

        return self.gradCAMs

    def set_gradhook(self,submodule=None):
        """
        set_gradhook

        Set hook on submodule for grad-CAM computation.

        inputs:
            submodule - (str or torch.nn.Module) name of submodule within self.model to hook OR submodule to hook directly
        """
        # define hook functions
        def getactivation(mod,input,output):
            """
            getactivation

            Copy activations to self.activations on forward pass.

            inputs:
                mod - (torch.nn.Module) module being hooked
                input - (tuple) inputs to submodule being hooked (ignore)
                output - (tensor) output of submodule being hooked (NOTE: untested for modules with multiple outputs)
            """
            self.activations = output

        def getgrad(mod,gradin,gradout):
            """
            getgrad

            Get gradients during hook.

            inputs:
                mod - (torch.nn.Module) module being hooked
                gradin - (tuple) inputs to last operation in mod (ignore)
                gradout - (tuple) gradients of class activation w.r.t. outputs of mod
            """
            self.grads = gradout[0]

        def getgrad_input(g):
            """
            getgrad_input

            Get gradient on backward pass when using inputs for CAMs.

            inputs:
                g - (torch.Tensor) tensor to set/store as gradient on backward pass
            """
            self.grads = g


        if submodule is None: # if using inputs as activation maps
            self.activations.requires_grad=True # make sure input grads are available
            self.bh = self.activations.register_hook(getgrad_input) # save gradient
        elif submodule.__class__.__name__ == 'str': # if using string name of submodule...
            for name,module in self.model._modules.items():
                if name == submodule:
                    self.fh = module.register_forward_hook(getactivation) # forward hook
                    self.bh = module.register_backward_hook(getgrad) # backward hook
        else: # if using the submodule itself
            self.fh = submodule.register_forward_hook(getactivation) # forward hook
            self.bh = submodule.register_backward_hook(getgrad) # backward hook

    def guidedhook(self,mod,gradin,gradout):
        """
        guidedhook

        Hook that applies a clamping operation to gradients for guided backpropagation.
        Modifies gradient during backprop by returning an altered gradin.

        inputs:
            mod - (torch.nn.Module) module being hooked
            gradin - (tuple) tuple of gradients of loss w.r.t inputs and parameters of mod.
                     Different for various layer types. For Conv2d layers, it's
                     (input.grad, mod.weight.grad, mod.bias.grad)
            gradout - (tuple) tuple of gradients of loss w.r.t. outputs of mod
        """
        if len(gradin) != 1:
            raise Exception('len(gradin) != 1. It should be equal to 1 for ReLU layers. Verify which modules you are hooking.')

        return (torch.clamp(gradin[0],min=0),)

    def deconvhook(self,mod,gradin,gradout):
        """
        deconvhook

        Hook that applies 'deconvolution' operation from https://arxiv.org/abs/1412.6806

        inputs:
            mod - (torch.nn.Module) module being hooked
            gradin - (tuple) tuple of gradients of loss w.r.t inputs and parameters of mod.
                     Different for various layer types. For Conv2d layers, it's
                     (input.grad, mod.weight.grad, mod.bias.grad)
            gradout - (tuple) tuple of gradients of loss w.r.t. outputs of mod
        """
        if len(gradin) != 1:
            raise Exception('len(gradin) != 1. It should be equal to 1 for ReLU layers. Verify which modules you are hooking.')

        return (torch.clamp(gradout[0],min=0),)

    def set_guidehooks(self,guided=False,deconv=False):
        """
        set_guidehooks

        Set hooks on ReLU modules for guided backprop.

        inputs:
            guided - (bool or torch.nn.Module list) use guided grad-CAM. Hooks all ReLU submodules if True, hooks given modules in list if list
            deconv - (bool) optionally use 'deconvolution' operation instead of guided backpropagation
        """
        hooktype = self.guidedhook
        if deconv == True:
            hooktype = self.deconvhook

        if guided is True:
            for name,module in self.model._modules.items():
                if module.__class__.__name__ == 'ReLU': # TODO: make this more elegant instead of creating a dummy Conv2d
                    h = module.register_backward_hook(hooktype)
                    self.guidehooks.append(h)
        elif guided.__class__.__name__ == 'list': # manually provided list of modules to hook
            for module in guided:
                h = module.register_backward_hook(hooktype)
                self.guidehooks.append(h)

        if self.verbose:
            print('len(self.guidehooks) =',len(self.guidehooks))
            print('guided backprop hooks set')

class Flatten(torch.nn.Module):
    """
    Flatten

    torch.flatten as a layer.
    """
    def __init__(self,start_dim=1):
        """
        __init__

        Constructor.

        inputs:
        start_dim - (bool) dimension to begin flattening at.
        """
        super(Flatten,self).__init__()
        self.start_dim = start_dim

    def forward(self,x):
        """
        forward

        Forward pass.
        """
        return x.flatten(self.start_dim)

def create_masked_image(x,cam,filename='examples/testimage.jpg'):
    """
    similar to show_cam_on_image from https://github.com/jacobgil/pytorch-grad-cam
    """
    mask = cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)
    mask = np.float32(mask) / 255
    mask = mask + np.float32(x)
    mask = mask / np.max(mask)
    if filename is not None:
        cv2.imwrite(filename,np.uint8(mask*255))

def preprocess_image(img):
    """
    Some VGG sample preprocessing copied from https://github.com/jacobgil/pytorch-grad-cam grad-cam.py
    """
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input
