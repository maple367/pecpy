import json
import logging
import multiprocessing
import os
import numpy as np
import time

from math import tanh
import imageio.v2 as imageio
from scipy.optimize import minimize

# region Optional imports

# noinspection PyBroadException
try:
    import pyfftw

    FFTW = True
    fftw_threads = multiprocessing.cpu_count()
    print("FFTW detected, enabling for speed up.")
except:
    FFTW = False
    print("Failed loading FFTW, falling back to numpy (about 5x slower).")


# endregion

# region Mathematical functions


def heavy(beta, eta, x):
    if beta is None:
        y = np.zeros_like(x)
        y[x > eta] = 1
        return y
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def d_heavy(beta, eta, x):
    with np.errstate(divide='ignore', over='ignore'):
        value = beta / pow(np.cosh(beta * (x - eta)), 2) / (np.tanh(beta * (1 - eta)) + tanh(beta * eta))
    return value


def n_heavy(beta, x, n):
    v = 0
    for i in range(1, n):
        v += heavy(beta, (1.0 * i) / n, x) / (n - 1)
    return v


def dn_heavy(beta, x, n):
    v = 0
    for i in range(1, n):
        v += d_heavy(beta, (1.0 * i) / n, x) / (n - 1)
    return v


def np_fft2(x):
    return np.fft.fft2(x)


def np_ifft2(x):
    return np.fft.ifft2(x)


def fftw2(x):
    return pyfftw.interfaces.numpy_fft.fft2(x, threads=fftw_threads, planner_effort="FFTW_MEASURE")


def ifftw2(x):
    return pyfftw.interfaces.numpy_fft.ifft2(x, threads=fftw_threads, planner_effort="FFTW_MEASURE")


def fft2(x):
    if FFTW:
        return fftw2(x)
    else:
        return np_fft2(x)


def ifft2(x):
    if FFTW:
        return ifftw2(x)
    else:
        return np_ifft2(x)


def conv2(x, fft_psf):
    fft = fft2(x)
    ff = fft * fft_psf
    return np.real(np.fft.fftshift(ifft2(ff)))


# endregion

# region Helper functions

def parse_int(s):
    try:
        return int(s)
    except:
        return None


def parse_image(image):
    if type(image) is np.array or type(image) is np.ndarray:
        return image if np.max(image) <= 1 else image / 255.0
    return imageio.imread(image) / 255.0


def rescale(im, scale):
    # Check if resize if necessary.
    if np.absolute(scale - 1.0) < 1e-6:
        return im, 1
    # Resize image.
    ix, iy = im.shape[:]
    #im2 = imageio.imresize(im * 255.0, (int(ix * scale), int(iy * scale))) / 255.0
    # Old code is deprecated.
    from PIL import Image
    im2 = np.array(Image.fromarray(im * 255.0).resize((int(ix * scale), int(iy * scale)))) / 255.0
    ix2, iy2 = im2.shape[:]
    s = (1.0 * ix) / ix2
    return im2, s


def dump_images(x, wd=None, template=None, beta=None):
    template = "proximity-PSF-{}.png" if template is None else template
    if beta is not None:
        template = template.format("{}-beta%04d" % beta)
    if wd is not None:
        template = os.path.join(wd, template)
    for key in x:
        imageio.imsave(template.format(key), x[key] * 255)


def make_psf(ix, iy, r):
    Y, X = np.meshgrid(range(iy), range(ix))
    psf = np.exp(-((X - ix / 2.0) ** 2 + (Y - iy / 2.0) ** 2) / r ** 2)
    psf /= psf.sum().sum()
    return psf, fft2(psf)


def setup_dir(dir, overwrite):
    if dir is not None:
        if not os.path.isdir(dir):
            os.makedirs(dir)
        elif not overwrite:
            raise ValueError("Directory exists: {}".format(dir))


def get_resource(name):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    return os.path.join(root, "resources", "{}".format(name))


# endregion

# region Classes

class EblSimulator:
    def __init__(self, ix, iy, alpha, eta):
        self.ix = ix
        self.iy = iy
        self.r = alpha
        self.psf, self.fft_psf = make_psf(ix, iy, alpha)
        self.eta = eta

    def predict(self, x_dose, beta=None):
        """
        Given a dose input, do exposure + development to determine x_exposed (the actual EBL expose taking into account
        forward scattering via the PFS) and x_phys (the final physical geometry, i.e. after development)
        :param x_dose: dose input
        :param beta: is not None, the development is smooth (lower beta, more smooth)
        :return: x_exposed, x_phys
        """
        x_exposed = conv2(x_dose, self.fft_psf)
        x_phys = heavy(beta, self.eta, x_exposed)
        return x_exposed, x_phys


# region Steppers

class FilterProjectionStepper:
    def __init__(self, alpha0, eta0s, dose_penalty=None):
        self.alpha0 = alpha0
        self.eta0s = eta0s
        self.dose_penalty = dose_penalty if dose_penalty is not None else 0
        # Inject later.
        self.psf0, self.fft_psf0 = None, None
        self.ix, self.iy = None, None
        self.beta = None
        self.ebl = None
        self.x_design = None

    def setup(self, x_design, ebl, scale):
        self.x_design = x_design
        self.ebl = ebl
        self.ix, self.iy = self.x_design.shape[:]
        self.psf0, self.fft_psf0 = make_psf(self.ix, self.iy, self.alpha0 / (scale if scale is not None else 1))

    def step(self, x, eta0=None):
        x = np.reshape(x, (self.ix, self.iy))
        eta0 = eta0 if eta0 is not None else self.eta0s[int(len(self.eta0s) / 2)]
        x0 = conv2(x, self.fft_psf0)
        x_dose = heavy(self.beta, eta0, x0)
        x_exposed, x_phys = self.ebl.predict(x_dose, self.beta)
        return x0, x_dose, x_exposed, x_phys

    def calc_f(self, x_dose, x_phys):
        f = ((x_phys - self.x_design) ** 2).sum().sum() / (self.ix * self.iy)
        dose_sum = x_dose.sum().sum() / (self.ix * self.iy)
        return f + dose_sum * self.dose_penalty

    def calc_g(self, x0, x_exposed, x_phys, eta0):
        term0 = 2 * (x_phys - self.x_design) / (self.ix * self.iy)
        term1 = (conv2(term0 * d_heavy(self.beta, self.ebl.eta, x_exposed), self.ebl.fft_psf) +
                 self.dose_penalty / (self.ix * self.iy))
        g = conv2(term1 * d_heavy(self.beta, eta0, x0), self.fft_psf0)
        return g.flatten()

    def f(self, x):
        fs = []
        for eta0 in self.eta0s:
            x0, x_dose, x_exposed, x_phys = self.step(x, eta0)
            fs.append(self.calc_f(x_dose, x_phys))
        return np.max(fs)

    def g(self, x):
        # Only one eta0, just calc g.
        if len(self.eta0s) == 1:
            x0, x_dose, x_exposed, x_phys = self.step(x, self.eta0s[0])
            return self.calc_g(x0, x_exposed, x_phys, self.eta0s[0])
        # Multiple eta0, calc f and g.
        fs, gs = [], []
        for eta0 in self.eta0s:
            x0, x_dose, x_exposed, x_phys = self.step(x, eta0)
            if len(self.eta0s) > 1:
                fs.append(self.calc_f(x_dose, x_phys))
            gs.append(self.calc_g(x0, x_exposed, x_phys, eta0))
        return gs[np.argmax(fs)]

# NOTE: Add more steppers here if needed.


# endregion

# region Solvers

class LBFGSB:
    def __init__(self, gtol, ftol, maxiter, disp=None, callback=None, bounds=None):
        self.gtol = gtol
        self.ftol = ftol
        self.disp = disp
        self.bounds = bounds
        self.maxiter = maxiter
        self.callback = callback

    def solve(self, x0, stepper):
        result = minimize(stepper.f, x0, jac=stepper.g, method="L-BFGS-B",
                          callback=lambda x: self.callback(x, stepper) if self.callback is not None else None,
                          bounds=[self.bounds] * (x0.shape[0] * x0.shape[1]) if self.bounds is not None else None,
                          options={"maxiter": self.maxiter, "disp": self.disp, "ftol": self.ftol, "gtol": self.gtol})
        return np.reshape(result['x'], x0.shape)

# Add more solver wrappers here if needed (could be e.g. MMA).


# endregion

# region Outputs
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
class DirectoryOutput:
    def __init__(self, directory, overwrite=False, template=None, modulo=None, betas=None, disp=None):
        self.directory = directory
        self.template = "{}.png" if template is None else template
        self.overwrite = overwrite
        self.modulo = modulo
        self.betas = betas
        self.count = 0
        self.disp = disp
        setup_dir(directory, overwrite)
        

    def info(self, info):
        with open(os.path.join(self.directory, "options.json"), 'w') as f:
            json.dump(info, f, cls=MyEncoder)

    def images(self, images, beta=None, force=None):
        if self.disp is not None and self.disp and beta is not None:
            print("Beta = {}".format(beta))
        if beta is not None:
            self.count += 1
            if (self.modulo is not None) and (self.count % self.modulo != 1) and (beta not in self.betas) and (
                    force is None or not force):
                return
        dump_images(images, self.directory, self.template, beta)

# Add more outputs here if needed (could be plotting extensions or something else)


# endregion

# endregion

# region Public interface

def develop_dose(dose, alpha, eta, design=None, output=None):
    # Write the applied options to output.
    if output is not None:
        output.info({"dose": dose, "alpha": alpha, "eta": eta, "design": design})
    # Read image.
    images = {"dose": parse_image(dose)}
    ix, iy = images["dose"].shape[:]
    # Do development.
    ebl = EblSimulator(ix, iy, alpha, eta)
    images["exposed"], images["phys"] = ebl.predict(images["dose"])
    # Calculate error if design is specified.
    if design is not None:
        images["design"] = parse_image(design)
        images["error"] = (images["phys"] - images["design"]) ** 2 / (ix * iy)
    # Output stuff.
    if output is not None:
        output.images(images)
    return images


def optimize_dose(x_design, alpha, eta, betas, stepper, solver, output=None, guess=None, scale=None, xtol=None):
    info = {}
    xtol = 5e-3 if xtol is None else xtol
    # Write the applied options to output.
    if output is not None:
        info["output"] = {**dict(output.__dict__), **{"type": output.__class__.__name__}}
        stepper_info = {**dict(stepper.__dict__), **{"type": stepper.__class__.__name__}}
        solver_info = {**dict(solver.__dict__), **{"type": solver.__class__.__name__}}
        solver_info["callback"] = None
        info = {**info, **{"design": x_design, "alpha": alpha, "eta": eta, "betas": betas, "stepper": stepper_info,
                           "solver": solver_info, "guess": guess, "scale": scale, "fftw": FFTW}}
        output.info(info)
    # Read image.
    x_design = parse_image(x_design)
    if scale is not None:
        x_design, s = rescale(x_design, scale)
        alpha /= s
    ix, iy = x_design.shape[:]
    if output is not None:
        output.images({"design": x_design})
    # Set initial value.
    guess = np.ones_like(x_design) * parse_int(guess) if parse_int(guess) is not None else guess
    x = np.copy(x_design) if guess is None else (parse_image(guess))
    # Setup ebl simulator.
    stepper.setup(x_design, EblSimulator(ix, iy, alpha, eta), scale)
    # Enable cache (significant performance gain).
    if FFTW:
        pyfftw.interfaces.cache.enable()
    # Run optimization.
    logging.info('::::: Optimization START (# design variables = {}) :::::'.format(ix * iy))
    tick = time.time()
    i = 0
    rms = np.inf
    while i < len(betas) - 1:
        if rms < xtol:
            break
        # Update beta.
        beta = betas[i]
        stepper.beta = beta
        # Solve the problem.
        x_prev = np.copy(x)
        x = solver.solve(x, stepper)
        # Check convergence.
        rms = np.sqrt(np.mean((x_prev - x) ** 2))
        # Output stuff.
        x0, dose, exposed, phys = stepper.step(x)
        images = {"x": x, "x0": x0, "dose": dose, "phys": phys, "error": (phys - x_design) ** 2 / (ix * iy)}
        if output is not None:
            output.images(images, beta)
        i += 1
    # Dump the last beta.
    stepper.beta = betas[-1]
    x0, dose, exposed, phys = stepper.step(x)
    images = {"x": x, "x0": x0, "dose": dose, "phys": phys, "design": x_design, "error": (phys - x_design) ** 2 / (ix * iy)}
    if output is not None:
        output.images(images, betas[-1], force=True)
    # Record info.
    tock = time.time()
    if output is not None:
        output.info({**info, **{"error": np.mean((phys - x_design) ** 2), "runtime": "{}".format(tock - tick)}})
    logging.info('::::: Optimization END :::::')
    return images

# endregion
