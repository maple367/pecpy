from pecpy.core import *
from pecpy.gdsutils import *

# Load resources.
cheese_mini = get_resource("cheese_mini.png")
# Set parameters.
alphas, etas = np.arange(10, 35, 5).tolist(), np.arange(0.1, 0.35, 0.05).tolist()
betas = [2.0 ** i for i in np.arange(0, 13)]
# File names.
home = "calibration"
file_name = "calibration"
element_name = "calibration array"
template = os.path.join(home, "alpha{}_eta{:.2f}")

# 1) Run optimizations.
for alpha in alphas:
    for eta in etas:
        output = DirectoryOutput(template.format(alpha, eta), overwrite=True)
        solver = LBFGSB(5e-7, 1e8 * np.finfo(float).eps, 50, disp=1)
        stepper = FilterProjectionStepper(10, [0.1])
        x = optimize_dose(cheese_mini, alpha, eta, betas, stepper, solver, output=output)

# 2) Convert to gds files (can take some time depending on the geometrical complexity of the optimized dose patterns).
for alpha in alphas:
    for eta in etas:
        img = misc.imread(os.path.join(template.format(alpha, eta), "dose-beta{}.png".format(int(betas[-1]))))
        lib = img2gds(img, levels=2, min_area=4, tolerance=1, center=True, smooth=(3, 1))
        with open("{}.gds".format(template.format(alpha, eta)), 'wb') as stream:
            lib.save(stream)

# 3) Create calibration array.
offset = [(int(0), int(0))]
x_spacing, y_spacing = 1000, 1000
# Load the files.
structs = [[]]
for alpha in alphas:
    structs.append([])
    for eta in etas:
        lib_file = "{}.gds".format(os.path.join(template.format(alpha, eta)))
        with open(lib_file, 'rb') as stream:
            lib = Library.load(stream)
        structs[-1].append(lib[0])
# Save the array.
lib = Library(5, str.encode('{}.DB'.format(file_name)), 1e-9, 0.001)
mother = Structure(str.encode("all"))
lib.append(mother)
lib.append(project_array2d(element_name, structs, x_spacing, y_spacing))
mother.append(SRef(str.encode(element_name), offset))
with open(os.path.join(home, "{}.gds".format(file_name)), 'wb') as stream:
    lib.save(stream)
