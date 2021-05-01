from matplotlib import pyplot as plt
import astropy.units as u

from ctapipe.image import toymodel
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.utils import datasets
from ctapipe.visualization import ArrayDisplay
from ctapipe.io import event_source
from tqdm import tqdm
from ctapipe.image import hillas_parameters, tailcuts_clean, HillasParameterizationError
from ctapipe.calib import CameraCalibrator
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

import shapely.geometry as sg
import shapely.ops as so



def find_nice_event():
    filename = datasets.get_dataset_path("gamma_test_large.simtel.gz")
    source = event_source(filename)
    calib = CameraCalibrator()
    
    for i, event in tqdm(enumerate(source)):
        # from IPython import embed; embed()
        # if i in [0, 1, 2, 3, 4, 39]:  # skip ugly events
        #     print(i, event.mc.energy)
        #     continue 
        subarray = event.inst.subarray
        calib(event)
        for tel_id in event.dl0.tels_with_data:
            # Camera Geometry required for hillas parametrization
            camgeom = subarray.tel[tel_id].camera
            # note the [0] is for channel 0 which is high-gain channel
            image = event.dl1.tel[tel_id].image
            # Cleaning  of the image
            cleaned_image = image.copy()
            # create a clean mask of pixels above the threshold
            cleanmask = tailcuts_clean(
                camgeom, image, picture_thresh=10, boundary_thresh=5, min_number_picture_neighbors=3
            )
            # set all rejected pixels to zero
            cleaned_image[~cleanmask] = 0

            # Calculate hillas parameters
            try:
                d = hillas_parameters(camgeom, cleaned_image)
            except HillasParameterizationError:
                pass  # skip failed parameterization (normally no signal)
            # from IPython import embed; embed()
            tel_name = event.inst.subarray.tel[tel_id].name
            if tel_name == 'LST' and d.r < 1 * u.m and d.intensity > 400:
                print(i, d.intensity, event.mc.energy)
                return tel_id, d, event, cleanmask


# importing data from avaiable datasets in ctapipe


# geom = CameraGeometry.from_name('LSTCam')


tel_id, d, event, cleanmask = find_nice_event()
geom = event.inst.subarray.tel[tel_id].camera
# from IPython import embed; embed()
image = event.dl1.tel[tel_id].image

size = plt.gcf().get_size_inches()
fig, [ax2, ax1] = plt.subplots(1, 2, figsize=(size[0], 2.4))

disp = CameraDisplay(geom, ax=ax1, norm=PowerNorm(0.5))
disp.image = image


selected_shapes = np.array(disp.pixels.get_paths())[cleanmask]

polygons = [sg.Polygon(s.to_polygons()[0]) for s in selected_shapes] 

new_shape = so.cascaded_union(polygons)
xs, ys = new_shape.exterior.xy
ax1.fill(xs, ys, alpha=1, fc='none', ec='white', lw=1)
# # from IPython import embed; embed()

# ax1.set_xticks([])
# ax1.set_yticks([])
ax1.set_xlabel('$x$-position / \\si{\\metre}')
ax1.set_ylabel('$y$-position / \\si{\\metre}')
ax1.set_title('')
# # ax.set_axis('off')
# mask = disp.image > 10
# disp.highlight_pixels(mask, linewidth=1, color='crimson')
waveforms = event.dl0.tel[tel_id].waveform.T
# from IPython import embed; embed()
m = (waveforms.max(axis=0) > 5) & (waveforms.max(axis=0) < 10)
selected_pixel = np.argmax(m)


mask = np.zeros_like(m).astype(np.bool)
mask[selected_pixel] = True
disp.highlight_pixels(mask, linewidth=1, color='C0', alpha=1)

t = np.arange(0, waveforms.shape[0], 1)
ax2.plot(t, waveforms[:, selected_pixel], color='C0', lw=1)
# from IPython import embed; embed()
ax2.set_xlabel('Time / \\si{\\nano\\second}')
ax2.set_ylabel('Voltage / a.u.')
ax2.set_title('')
ax2.set_xlim([0, 29])

plt.tight_layout(pad=0, rect=(0.0, 0, 1.007, 1))
plt.subplots_adjust(wspace=0.35)
plt.savefig('build/preprocessing.pdf')

with open('build/preprocessing_energy.txt', 'w') as f:
    energy = event.mc.energy.to_value('TeV')
    f.write(f'\\SI{{{energy:.2f}}}{{TeV}}')

with open('build/preprocessing_multi.txt', 'w') as f:
    multi = len(event.dl0.tels_with_data)
    f.write(f'{multi}')