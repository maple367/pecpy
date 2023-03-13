import cv2
import itertools

from PIL import Image, ImageDraw
from gdsii.library import Library
from gdsii.structure import *
from gdsii.elements import *
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from scipy.signal import savgol_filter

import numpy as np
import shapely.affinity as aff


# region Rendering of gds library to image.

def gds2img(n, m, lib, decenter=True, dose_max=1000):
    """
    Convert gds library to an image
    :param lib: gds library (or path to gds library file)
    :param n: width of image
    :param m: height of images
    :param decenter: shift geometries from (0, 0) to center of canvas (n/2, m/2)
    :param dose_max: maximum dose value in gds file
    :return: grey scale image
    """
    return bnds2img(n, m, gds2bnds(lib), dose_max, decenter)


def gds2bnds(lib):
    """
    Convert gds library to an image
    :param lib: gds library (or path to gds library file)
    :return: bnds
    """
    # Load the library.
    if type(lib) is not Library:
        with open(lib, 'rb') as stream:
            lib = Library.load(stream)
    # Extract paths.
    bnds = []
    for struc in lib:
        for elem in struc:
            # For now, only boundary elements are supported.
            if type(elem) is not Boundary:
                continue
            bnds.append(elem)
    return bnds


def bnds2img(n, m, bnds, dose_max, decenter=True):
    """
    Convert a list of paths to an image.
    :param n: width of image
    :param m: height of images
    :param bnds: boundaries to paint.
    :param dose_max: maximum dose value in gds file
    :param decenter: shift geometries from (0, 0) to center of canvas (n/2, m/2)
    :return: grey scale image
    """
    img = np.zeros((n, m))
    for bnd in bnds:
        # Decenter if needed.
        path = bnd.xy
        if decenter:
            path = [(item[0] + m / 2, item[1] + n / 2) for item in path]
        # Paint on the canvas.
        img_elem = Image.new('L', (m, n), 0)
        ImageDraw.Draw(img_elem).polygon(path, outline=1, fill=1)
        img_elem = np.array(img_elem)
        img[img_elem != 0] = (np.array(img_elem[img_elem != 0]) * (bnd.data_type / dose_max * 255))
    return img.astype(np.uint8)


# endregion

# region Parsing of image to gds file.

def img2gds(image, levels=None, min_area=None, tolerance=None, center=False, scale=None, smooth=None, dose_max=1000):
    """
    Convert an image to a gds library.
    :param image: image to convert
    :param levels: number of threshold levels (minimum 2, i.e. black and white)
    :param min_area: minimum feature size, features with area < min_size are skipped
    :param tolerance: if not None, paths are simplified to this tolerance (see shapely documentation for details)
    :param center: if True, the geometry coordinates are centered
    :param scale: if not None, the traced coordinates are scaled
    :param smooth: if not None, specifies a tuple of
    :param dose_max: maximum dose value written to gds file
    :return: gds library
    """
    lib = Library(5, b'img2gds.DB', 1e-9, 0.001)
    lib.append(
        img2struct('img2gds', image, dose_max, levels=levels, min_area=min_area, tolerance=tolerance, center=center,
                   scale=scale, smooth=smooth))
    return lib


def img2struct(tag, image, dose_max, levels=None, min_area=None, tolerance=None, center=False, scale=None, smooth=None):
    """
    Convert an image to a gds library.
    :param image: image to convert
    :param dose_max: maximum dose value assigned to boundaries
    :param levels: number of threshold levels (minimum 2, i.e. black and white)
    :param min_area: minimum feature size, features with area < min_size are skipped
    :param tolerance: if not None, paths are simplified to this tolerance (see shapely documentation for details)
    :param center: if True, the geometry coordinates are centered
    :param scale: if not None, the traced coordinates are scaled
    :param smooth: if not None, specifies a tuple of
    :return: gds structure
    """
    struct = Structure(str.encode(tag))
    for bnd in img2bnds(image, dose_max, levels, min_area, tolerance, center, scale, smooth):
        struct.append(bnd)
    return struct


def img2bnds(image, dose_max, levels=None, min_area=None, tolerance=None, center=False, scale=None, smooth=None):
    """
    Convert an image to a list of geometries
    :param image: image to convert
    :param dose_max: maximum dose value assigned to boundaries
    :param levels: number of threshold levels (minimum 2, i.e. black and white)
    :param min_area: minimum feature size, features with area < min_size are skipped
    :param tolerance: if not None, paths are simplified to this tolerance (see shapely documentation for details)
    :param center: if True, the geometry coordinates are centered
    :param scale: if not None, the traced coordinates are scaled
    :param smooth: if not None, specifies a tuple of
    :return: list of boundaries
    """
    return polys2bnds(img2polys(image, levels, min_area, tolerance, center, scale, smooth), dose_max)


def img2polys(image, levels=None, min_area=None, tolerance=None, center=False, scale=None, smooth=None):
    """
    Convert an image to a list of geometries
    :param image: image to convert
    :param levels: number of threshold levels (minimum 2, i.e. black and white)
    :param min_area: minimum feature size, features with area < min_size are skipped
    :param tolerance: if not None, paths are simplified to this tolerance (see shapely documentation for details)
    :param center: if True, the geometry coordinates are centered
    :param scale: if not None, the traced coordinates are scaled
    :param smooth: if not None, specifies a tuple of
    :return: list of boundaries
    """
    polygon_lists = []
    exposed = None
    ix, iy = image.shape[0], image.shape[1]
    values = sorted(np.unique(image / 255.0).tolist()) if levels is None else np.linspace(0, 1, levels, endpoint=True)
    levels = 2 if levels is None else levels
    for i in reversed(range(0, len(values) - 1)):
        # Trace contours (TODO: Can this be done without CV2?).
        lower = np.array([255 * values[i] + (255 / levels)])
        upper = np.array([255 + (255 / levels)])
        mask = cv2.inRange(image, lower, upper)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [np.reshape(c, (c.shape[0], c.shape[-1])) for c in contours]
        # Filter out contours with < 6 points.
        contours = [c for c in contours if c.size > 6]
        # Smooth contours if needed.
        if smooth is not None:
            contours = [smooth_contour(c, smooth[0], smooth[1]) for c in contours]
        # Create polygons.
        polygons = [Polygon(c) for c in contours]
        # Apply tolerance to decrease the number of points in each polygon.
        if tolerance is not None:
            polygons = [polygon.simplify(tolerance) for polygon in polygons]
        # Filter out invalid polygon.
        polygons = [polygon if polygon.is_valid else polygon.buffer(0) for polygon in polygons]
        # Assemble polygons.
        polygons = assemble_polygons(polygons)
        # Subtract already-exposed areas.
        if exposed is not None:
            for entry in [exposed] if hasattr(exposed, "exterior") else exposed.geoms:
                polygons = [polygon.difference(entry) for polygon in polygons]
        # Filter out polygons with area < min_air.
        if min_area is not None:
            polygons = [(MultiPolygon([geom for geom in polygon.geoms if geom.area > min_area]) if hasattr(polygon,
                                                                                                           "geoms") else polygon)
                        for polygon in polygons if polygon.area > min_area]
        # Update exposed region.
        exposed = unary_union(polygons if exposed is None else polygons + [exposed])
        # Transform and collect polygons.
        polygon_lists.append([])
        for polygon in polygons:
            entries = [polygon] if hasattr(polygon, "exterior") else polygon.geoms
            for entry in entries:
                if entry.exterior is None:
                    continue
                entry = aff.scale(entry, scale, scale, origin=(ix / 2, iy / 2)) if scale is not None else entry
                entry = aff.translate(entry, -ix / 2, -iy / 2) if center else entry
                polygon_lists[i].append(entry)
    return polygon_lists


def polys2bnds(polygon_lists, dose_max):
    boundaries = []
    levels = len(polygon_lists) + 1
    for i, polygons in enumerate(polygon_lists):
        bnds_i = []
        value_i = (float(i + 1) / (levels - 1))  # if value is None else value
        for polygon in polygons:
            bnd_i = [(int(item[0]), int(item[1])) for item in flatten_polygon(polygon)]
            bnds_i.append(Boundary(i, int(value_i * dose_max), bnd_i))
        boundaries.extend(reversed(bnds_i))
    return boundaries


def assemble_polygons(polygons):
    """
    From a number of (potentially overlapping) polygons and holes, a single polygon is assembled. First for loop
    determines if the structure is a hole or a feature. Second for loop calculates the resulting polygon
    :param polygons: input polygons
    :return: a single polygon
    """
    hole_mask = [False] * len(polygons)
    for j, polygon in enumerate(polygons):
        for k, other in enumerate(polygons):
            if k == j:
                continue
            if polygon.within(other):
                if hole_mask[j]:
                    hole_mask[j] = False
                else:
                    hole_mask[j] = True
    for j, polygon in enumerate(polygons):
        if hole_mask[j]:
            for k, other in enumerate(polygons):
                if k == j:
                    continue
                if polygon.within(other):
                    polygons[k] = polygons[k].difference(polygon)
    return list(itertools.compress(polygons, [not b for b in hole_mask]))


def smooth_contour(contour, window=None, order=None):
    """
    Smooth a contour using the savgol filter. For correct smoothing at the end, the contour wrapped at the end.
    :param contour: contour to smooth
    :param window: smoothing window
    :param order: smoothing order
    :return: smoothed contour
    """
    x, y = contour[:, 0], contour[:, 1]
    n, n2 = x.shape[0], int(x.shape[0] / 2)
    x, y = np.concatenate((x, x)), np.concatenate((y, y))
    x, y = savgol_filter(x, window, order), savgol_filter(y, window, order)
    x = np.concatenate((x[n:n + n2], x[n2:n]))
    y = np.concatenate((y[n:n + n2], y[n2:n]))
    return np.column_stack((x, y))


def flatten_polygon(polygon):
    """
    The gds format works only with "simple" polygons, i.e. a polygon must be represented by a coordinate sequence.
    If the polygon is complex, i.e. it has holes, a single coordinate sequence including the holes must be created
    (the holes are thus "attached" to the exterior boundary). This method collapses a geometry object into a such
    representation. This operation might be VERY expensive (depending on the complexity of involved polygons).
    :param polygon: polygon to convert
    :return: connected boundary
    """
    shell, holes = polygon.exterior.coords, [item.coords for item in polygon.interiors]
    # Simple polygon, just return.
    if len(holes) == 0:
        return shell
    # Complex polygon, subtract holes.
    polygons = [shell] + holes
    link_map = {i: [] for i in range(len(polygons))}
    obstacles = [LineString(polygon) for polygon in polygons]
    # Loop over polygons which are to-be-connected.
    for i in range(1, len(polygons)):
        connected = False
        # Loop over polygons to which the connection is made.
        for j in range(0, len(polygons)):
            if i == j:
                continue
            link = check_connection(polygons[j], polygons[i], obstacles)
            if link is not None:
                link_map[j].append([link, i])
                connected = True
                break
        if not connected:
            raise ValueError("No connection found.")
    return unfold_connections(0, polygons, link_map, 0)


def check_connection(poly1, poly2, obstacles):
    """
    Check if it is possible to connect poly1 to poly2 without crossing the obstacles. If multiple connection are
    possible, the shortest possible connection is returned. As ALL connection are considered, this operation might be
    very expensive depending on the resolution of poly1 and poly, in particular if no connection is possible.
    :param poly1: polygon to be connected
    :param poly2: polygon to which poly1 should be connected
    :param obstacles: obstacles
    :return: connection indices if connection is possible, else None
    """
    n, m = len(poly1), len(poly2)
    # Check which points are close.
    costs = np.empty((n, m))
    for i, coord1 in enumerate(poly1):
        for j, coord2 in enumerate(poly2):
            costs[i, j] = (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2  # squared distance
    costs = np.argsort(costs.flatten())
    # Check if a direct connection is possible.
    for k in costs:
        i, j = np.unravel_index(k, (n, m))
        i, j = int(i), int(j)
        line = LineString([poly1[i], poly2[j]])
        intersects = 0
        for obstacle in obstacles:
            if line.intersects(obstacle):
                intersects += 1
        if intersects > 2:
            continue
        return [i, j]
    return None


def unfold_connections(poly_idx, polygons, link_map, offset):
    """
    Unfold the connection of polygons[poly_idx] recursively.
    :param poly_idx: index of polygon
    :param polygons: list of all polygons
    :param link_map: map of links between polygons
    :param offset: offset to apply in order to align connection points
    :return: a coordinate list representing polygons[poly_idx] along with all connected polygons
    """
    # Then, connect them.
    links = link_map[poly_idx]
    links = sorted(links, key=lambda key: key[0][0]) if len(links) > 0 else links
    current = polygons[poly_idx][offset:] + polygons[poly_idx][:offset] + [polygons[poly_idx][offset]]
    result = []
    last_idx = 0
    for entry in links:
        # Add part of current polygon.
        next_idx = entry[0][0] - offset
        result.extend(current[last_idx: next_idx + 1])
        last_idx = next_idx
        # Add connected polygon.
        unfolded = unfold_connections(entry[1], polygons, link_map, entry[0][1])
        result.extend(unfolded)
    result.extend(current[last_idx:])
    return result


# endregion

# region Util methods

def project_array2d(name, structs, dx, dy):
    struct = Structure(str.encode(name))
    for i, elem in enumerate(structs):
        for j, item in enumerate(elem):
            struct.extend(translate(item, dx * j, dy * i))
    return struct


def translate(obj, dx, dy):
    if hasattr(obj, "__len__"):
        for item in obj:
            translate(item, dx, dy)
    elif hasattr(obj, "xy"):
        obj.xy = [(int(item[0] + dx), int(item[1] + dy)) for item in obj.xy]
    return obj


def map_attribute(obj, mapping, attr):
    if hasattr(obj, "__len__"):
        for item in obj:
            map_attribute(item, mapping, attr)
    elif hasattr(obj, attr):
        setattr(obj, attr, mapping[getattr(obj, attr)])
    return obj

# endregion
