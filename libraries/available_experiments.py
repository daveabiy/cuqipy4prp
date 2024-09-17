from skimage.data import shepp_logan_phantom, astronaut, camera, horse
from skimage.color import rgb2gray
from libraries.fresnel_propagator import *
from skimage.draw import polygon
import numpy as np

def one_matrix(n, m, i, j, r=0  , shape='rectangle'):
    A = np.zeros((n, m))
    A[i, j] = 1
    
    if r == 0:
        r = min(n, m) // 10
    
    if shape == 'rectangle':
        A[i-r//2:i+r//2, j-r//2:j+r//2] = 1
    elif shape == 'triangle':
        for x in range(n):
            for y in range(m):
                if np.abs(x-i) + np.abs(y-j) < r:
                    A[x, y] = 1
    elif shape == 'circle':
        y, x = np.ogrid[:n, :m]
        mask = (x - j)**2 + (y - i)**2 < r**2
        A[mask] = 1
    elif shape == 'gaussian circle':
        y, x = np.ogrid[:n, :m]
        A = np.exp(-((x - j)**2 + (y - i)**2) / (2 * r**2))    
    return A

def P_one_matrices(P=10, n=128, m=128, r='any', shape='any', add_all = True, numbers = 1):
    shapes = ['rectangle', 'triangle', 'circle']
    if shape == 'any':
        random_shapes = [np.random.choice(shapes) for i in range(P)]
    else:
        random_shapes = [shape for i in range(P)]
    if numbers == 1:
        if r == 'any':
            A = [one_matrix(n, m, np.random.randint(0, n), np.random.randint(0, m), np.random.randint(0, min(n, m)//P + 20), shape=random_shapes[i]) for i in range(P)]
        else:
            A = [one_matrix(n, m, np.random.randint(0, n), np.random.randint(0, m), r, shape=shape) for _ in range(P)]
        if add_all:
            A_sum = np.sum(A, axis = 0)
            return A_sum
        else:
            return A
    else:
        A = [P_one_matrices(P, n, m, r, shape, add_all) for i in range(numbers)]
        return A

class available_experiments():
    def __init__(self, **kwargs):
        args = {
            'phase': None, 
            'attenuation': None,
            'abs_ratio': 1,
            'fresnel_number': 0.0001,
            'mode': 'reflect',
            'ground_atten_transform_type': 'reshape',
            'positive_phase': 'relu_inverted',
            'positive_attenuation': 'gelu',
            'value': 'min',
            'transform_type': 'reshape',
            'add_noise': False,
            'noise_factor': 0.0036,
            'no_of_noises': 5,
            'noise_type': 'gaussian',
            'seed': 42,
            'ground_transform_type': 'reshape',
            'dict': None,
            'remove_extreme': False,
            'dims': None,
            'cut': None,
            'horizontally': True,
            'vertically': True,
            'downsampling_factor': 1
        }
        kwargs = join_dict( kwargs, args)
        for key, value in kwargs.items():
            setattr(self, key, value)
            args[key] = value
                
    def propagate_others(self, phase = None, attenuation = None, downsampling_factor = 4, abs_ratio = 1, fresnel_number = 0.0001, mode = 'reflect', ground_atten_transform_type = 'reshape', positive_phase = 'relu_inverted', positive_attenuation = 'gelu', value = 'min', transform_type = 'reshape', add_noise =False, noise_factor = 0.0036, no_of_noises = 5, noise_type = 'gaussian', seed = 42, ground_transform_type = 'reshape', dict = None, remove_extreme = False, **kwargs):
        assert phase is not None and attenuation is not None, "phase and attenuation must be in kwargs"
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if type(phase) is not torch.Tensor:
            if len(phase.shape) == 3:
                phase = rgb2gray(phase)
            if len(attenuation.shape) == 3:
                attenuation = rgb2gray(attenuation)
            dim = (int(phase.shape[0]//downsampling_factor), int(phase.shape[1]//downsampling_factor)) if 'dim' not in kwargs.keys() else kwargs['dim']
            if dim is None:
                dim = (kwargs['dim'], kwargs['dim']) if type(kwargs['dim']) is int else (kwargs['dim'][0], kwargs['dim'][1])
        else:
            dim = (int(phase.shape[2]//downsampling_factor), int(phase.shape[3]//downsampling_factor)) if 'dim' not in kwargs.keys() else kwargs['dim']
        
        fresnel_number = fresnel_number * downsampling_factor**2
        phase = pos_neg_image(transform(phase, ground_transform_type), positive_phase, remove_extreme=remove_extreme)
        attenuation = pos_neg_image(transform(attenuation, ground_atten_transform_type), remove_extreme=remove_extreme)

        phase = T.Resize(size = dim, antialias=True)(phase)
        attenuation = T.Resize(size = dim, antialias=True)(attenuation) * abs_ratio
    
        simulation_sphere = {
            'experiment_name': 'simulation spheres',
            'phase': phase,
            'attenuation': attenuation,
            'fresnel_number': fresnel_number,
            'downsampling_factor': 1,
            'mode': mode,
            'pad' : 2,
            'abs_ratio': abs_ratio,
            'ground_truth': phase,
            'ground_attenuation': attenuation,
            'transform_type': transform_type,
            'ground_transform_type': ground_transform_type,
            'ground_atten_transform_type': ground_atten_transform_type,
            'positive_phase': positive_phase,
            'positive_attenuation': positive_attenuation,
            'value': value,
            'add_noise': add_noise,
            'noise_factor': noise_factor,
            'no_of_noises': no_of_noises,
            'seed': seed,
            'noise_type': noise_type,
            'cut': None if 'cut' not in kwargs.keys() else kwargs['cut'],
            'horizontally': True if 'horizontally' not in kwargs.keys() else kwargs['horizontally'],
            'vertically': True if 'vertically' not in kwargs.keys() else kwargs['vertically'],
        }
        prop = Fresnel_propagation(**simulation_sphere)
        simulation_sphere['image'] = prop.image
        if add_noise:
            if noise_type == 'random':
                noise_type = np.random.choice(['gaussian', 'poisson', 'speckle', None], 1)[0]
                noise_factor = np.random.uniform(0, 0.2, 1).item()
                
            if noise_type == 'gaussian':
                simulation_sphere['image'] = torch_noise_gaussian(prop.image, noise_factor)
            elif noise_type == 'poisson':
                simulation_sphere['image'] = torch_noise_poisson(prop.image, noise_factor, torch.Generator(device='cpu').manual_seed(seed))
            elif noise_type == 'speckle':
                simulation_sphere['image'] = torch_noise_speckle(prop.image, noise_factor)
                
            noise_type = 'gaussian' if noise_type is None else noise_type
            simulation_sphere['noise_factor'] = noise_factor
            simulation_sphere['noise_type'] = noise_type
            simulation_sphere['add_noise'] = True if noise_type is not None else False
        else:
            simulation_sphere['image'] = prop.image        
        simulation_sphere['transformed_images'] = transform(simulation_sphere['image'], transform_type)
        simulation_sphere['wavefield'] = prop.wavefield
        simulation_sphere['fresnel_factor'] = prop.fresnel_factor
        return join_dict(dict, simulation_sphere)

    def propagate_with_self(self):
        assert self.phase is not None and self.attenuation is not None, "phase and attenuation must be in kwargs"
        return self.propagate_others(phase = self.phase, attenuation = self.attenuation, downsampling_factor = self.downsampling_factor, abs_ratio = self.abs_ratio, fresnel_number = self.fresnel_number, mode = self.mode, ground_atten_transform_type = self.ground_atten_transform_type, positive_phase = self.positive_phase, positive_attenuation = self.positive_attenuation, value = self.value, transform_type = self.transform_type, add_noise = self.add_noise, noise_factor = self.noise_factor, no_of_noises = self.no_of_noises, noise_type = self.noise_type, seed = self.seed, ground_transform_type = self.ground_transform_type, dict = self.__dict__, remove_extreme = self.remove_extreme)
   