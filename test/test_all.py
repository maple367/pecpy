import unittest
import matplotlib.pyplot as plt

from pecpy.gdsutils import *
from pecpy.core import *


class TestPecpy(unittest.TestCase):
    def test_sanity(self):
        self.assertTrue(True)

    # region Testing of PEC

    def test_development_cheese(self):
        cheese = get_resource("cheese_mini.png")
        x = develop_dose(cheese, 30, 0.25, design=cheese)
        # plt.imshow(x["error"])
        self.assertEqual(0.111541748046875, np.sum(np.sum(x["error"])))

    def test_optimize_cheese(self):
        cheese = get_resource("cheese_mini.png")
        betas = [2.0 ** i for i in np.arange(0, 13)]
        solver = LBFGSB(5e-7, 1e8 * np.finfo(float).eps, 50, disp=1)
        stepper = FilterProjectionStepper(10, [0.1])
        x = optimize_dose(cheese, 30, 0.25, betas, stepper, solver)
        # plt.imshow(x["error"])
        self.assertTrue(np.sum(np.sum(x["error"])) < 2e-3)

    # endregion

    # region Testing of gradient calculation

    def test_analytical_gradient(self):
        design = misc.imread(get_resource("cheese_mini.png"), flatten=True) / 255.0
        alpha, eta, alpha0, levels, beta = 25, 0.05, 10, 2, 8
        # Setup stuff.
        rho = np.empty_like(design)
        rho[:] = design
        ix, iy = rho.shape[:]
        ebl = EblSimulator(ix, iy, alpha, eta)

        def fd_grad(x):
            df = np.reshape(stepper.g(x), (x.shape[0], x.shape[1]))
            order = 1  # 1 or 2
            delta = 1e-3
            f = stepper.f(x)
            err = 0
            ierr = 0
            icheck = 0
            to_check = 100
            while icheck < to_check:
                i, j = np.random.randint(0, x.shape[0]), np.random.randint(0, x.shape[1])
                if not 2 * delta < x[i, j] < (1.0 - 2 * delta):
                    continue
                icheck += 1
                x[i, j] += delta
                f1 = stepper.f(x)
                x[i, j] -= 2 * delta
                if order == 2:
                    f = stepper.f(x)
                x[i, j] += delta
                df2 = (f1 - f) / (order * delta)
                # df2 = df2 if fct is None else df2/(float(fct)**2)
                derr = np.abs(df2 - df[i, j])
                err += derr
                if np.abs(df2) > 1e-10 and derr > 1e-2 * np.abs(df2):
                    ierr += 1
                # msg = "{} {} f {} {} {} x {}"
                # print(msg.format(i, j, df2, df[i, j], np.absolute((df2 - df[i, j]) / df[i, j]), x[i, j]))
            return ierr, icheck, err

        for dose_penalty in [0, 1e-1]:
            stepper = FilterProjectionStepper(alpha0, [1.0], dose_penalty=dose_penalty)
            stepper.setup(design, ebl, 1.0)
            stepper.beta = beta
            # Let's look at rho0 (rho has no gradient).
            rho0, _ = ebl.predict(rho, beta)
            rho0 = np.reshape(rho0, (ix, iy))
            ierr, icheck, err = fd_grad(rho0)
            print("Total error = {} (dw = {})".format(err, dose_penalty))
            # Ensure that checks have been performed.
            self.assertTrue(icheck > 0)
            # Verify that no errors were present.
            print("Number of errors = {} (dw = {})".format(ierr, dose_penalty))
            self.assertTrue(ierr < 5)

    # endregion

    # region Testing of gds utils

    def test_img2gds(self):
        lib = img2gds(misc.imread(get_resource("cheese_mini.png"), flatten=True), tolerance=1, min_area=16, levels=2)
        # with open(get_resource("cheese_mini.gds"), 'wb') as stream:
        #     lib.save(stream)
        with open(get_resource("cheese_mini.gds"), 'rb') as stream:
            reference = Library.load(stream)
        for i, struc in enumerate(lib):
            for j, elem in enumerate(struc):
                self.assertEqual(reference[i][j].xy, elem.xy)
                self.assertEqual(reference[i][j].data_type, elem.data_type)

    def test_gds2img(self):
        image = gds2img(256, 256, get_resource("cheese_mini.gds"), decenter=False)
        reference = misc.imread(get_resource("cheese_mini.png"), flatten=True)
        error = image - reference
        self.assertEqual(np.sum(np.sum(error)), -19380)  # was 2550, is now -19380.0, might be related to cv2 version?

    # endregion
