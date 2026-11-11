
import os
import sys
import time
import numpy as np
import shutil
import exoclock

from astropy.io import fits as pf

from hops.hops_tools.files import find_fits_files, get_fits_data_and_header, open_fits, save_fits
from hops.hops_tools.files import open_dict, save_dict
from hops.hops_tools.image_analysis import image_burn_limit
from hops.hops_tools.reduction import combine_frames, reduce_dark_frame, reduce_flat_frame, stack_frames_calc
from hops.hops_tools.reduction import compute_master_flat, reduce_science_frame, bin_and_crop_science_frame
from hops.hops_tools.reduction import add_image_statistics_to_header, add_timing_and_coordinates_to_header
from hops.hops_tools.reduction import mean_std_from_median_mad, get_saturation
from hops.application_windows import MainWindow


class ReductionWindow(MainWindow):

    def __init__(self, log):

        MainWindow.__init__(self, log, name='HOPS - Reduction', position=2)

        self.compress_reduced = self.log.get_param('compress_reduced')

        self.file_info = []

        self.bias_files = find_fits_files(self.log.get_param('bias_files'))
        self.bias_frames = []
        self.bias_exp_time = []
        self.bias_counter = 0

        self.dark_files = find_fits_files(self.log.get_param('dark_files'))
        self.dark_frames = []
        self.dark_counter = 0

        self.darkf_files = find_fits_files(self.log.get_param('darkf_files'))
        self.darkf_frames = []
        self.darkf_counter = 0

        self.flat_files = find_fits_files(self.log.get_param('flat_files'))
        self.flat_frames = []
        self.flat_counter = 0

        self.science_files = find_fits_files(self.log.get_param('observation_files'))
        self.all_frames = {}
        self.science_counter = 0
        self.ref_stars = []
        self.psf = 10

        location_string = self.log.get_param('location').split(' ')
        self.observatory = exoclock.Observatory(exoclock.Degrees(location_string[0]), 
                                                exoclock.Degrees(location_string[1]))
        ra_dec_string = self.log.get_param('target_ra_dec').split(' ')
        self.target = exoclock.FixedTarget(exoclock.Hours(ra_dec_string[0]), exoclock.Degrees(ra_dec_string[1]))

        self.filter = self.log.get_param('filter')

        test_fits_name = self.science_files[0]
        test_fits_data, test_fits_header = self.get_fits_data_and_header(test_fits_name)

        t0 = time.time()
        _ = mean_std_from_median_mad(test_fits_data)
        self.fr_time = int(2000 * (time.time()-t0))

        self.progress_figure = self.FitsWindow(fits_data=test_fits_data, fits_header=test_fits_header,
                                               input_name=self.science_files[0])
        self.progress_bias = self.Progressbar(task="Loading bias frames")
        self.progress_dark = self.Progressbar(task="Loading dark frames")
        self.progress_darkf = self.Progressbar(task="Loading dark-flat frames")
        self.progress_flat = self.Progressbar(task="Loading flat frames")
        self.progress_science = self.Progressbar(task="Reducing science frames and calculating statistics")
        self.progress_science_loop = self.CheckButton(text='Show all frames', initial=0)
        self.progress_stacking = self.Progressbar(task="Stacking frames")

        setup_window = [
            [[self.progress_figure, 0, 2]]
        ]

        if len(self.bias_files) > 0:
            setup_window.append([[self.progress_bias, 0, 2]])
        if len(self.dark_files) > 0:
            setup_window.append([[self.progress_dark, 0, 2]])
        if len(self.darkf_files) > 0:
            setup_window.append([[self.progress_darkf, 0, 2]])
        if len(self.flat_files) > 0:
            setup_window.append([[self.progress_flat, 0, 2]])

        setup_window += [
            [[self.progress_science, 0], [self.progress_science_loop, 1]],
        ]

        t_eff_setting = self.log.get_param('effective_exposure')
        if t_eff_setting not in ["OFF", "off", "Off", None, ""]:
            setup_window.append([[self.progress_stacking, 0, 2]])

        setup_window += [
            [[self.Button(text='STOP REDUCTION & RETURN TO MAIN MENU', commands=[self.trigger_exit]), 0, 2]],
            []
        ]

        self.setup_window(setup_window)

        self.set_close_button_function(self.trigger_exit)

    def get_fits_data_and_header(self, fits_name):

        fits_data, fits_header = get_fits_data_and_header(fits_name)

        if self.file_info == []:
            self.file_info = [len(fits_data[0]), len(fits_data), fits_header['BITPIX']]
        else:
            if (len(fits_data[0]) != self.file_info[0] or self.file_info[1] != len(fits_data) or
                    self.file_info[2] != fits_header['BITPIX']):
                self.showinfo(
                    'Inconsistent image size',
                    'File {0} has different size from the other files in the dataset:\n'
                    'x-size: {1} pixels\ny-size: {2} pixels\n{3}-bit\nReduction will terminate, '
                    'please check your files!'.format(
                        fits_name, len(fits_data[0]), len(fits_data), fits_header['BITPIX'])
                )
                self.show()
                self.exit = True
                self.after(self.reduce_science)

        return fits_data, fits_header

    # define functions

    def run_reduction(self):

        self.progress_figure.adjust_size()

        reduction_complete = self.log.get_param('reduction_complete')
        stacking_complete = self.log.get_param('stacking_complete')
        t_eff_setting = self.log.get_param('effective_exposure')
        stacking_enabled = t_eff_setting not in ["OFF", "off", "Off", None, ""]

        if reduction_complete:
            if stacking_enabled and stacking_complete:
                status_msg = 'Reduction and stacking have both been completed.'
            elif stacking_enabled and not stacking_complete:
                status_msg = 'Reduction is complete but stacking has not been run yet.'
            else:
                status_msg = 'Reduction has been completed.'

            if self.askyesno('Overwrite files',
                             f'{status_msg}\n\nDo you want to re-run the full reduction from scratch?'):
                self.log.set_param('reduction_complete', False)
                self.log.set_param('stacking_complete', False)
                self.log.save_local_log()
            else:

                if stacking_enabled and not stacking_complete:

                    if self.askyesno('Stack existing reduced data',
                                     f'Stacking is set to an effective exposure of {t_eff_setting} seconds.\n\n'
                                     'Do you want to stack the existing reduced data?'):
                        print("\nStacking existing reduced data...")
                        # try:
                        self.stack()
                        self.showinfo('Stacking Complete', 'Stacking completed successfully!\n\n'
                                                           'Stacked frames are in: REDUCED_DATA/')
                        # except Exception as e:
                        #     print(f"Error during stacking: {e}")
                        #     self.showinfo('Stacking Error', f'Error occurred during stacking:\n{e}')

                self.log.set_param('proceed', True)
                self.close()
                return None

        if not self.log.get_param('reduction_complete'):

            if not os.path.isdir(self.log.reduction_directory):
                os.mkdir(self.log.reduction_directory)
            else:
                shutil.rmtree(self.log.reduction_directory)
                os.mkdir(self.log.reduction_directory)

            for path in [
                self.log.all_frames_path,
                self.log.all_stars_path,
                self.log.plate_solutions_path,
            ]:
                try:
                    os.remove(path)
                except:
                    pass

            self.after(self.get_bias)

        else:
            self.close()

    # reduction routines

    def get_bias(self):

        if self.exit or len(self.bias_files) == 0:
            self.after(self.get_master_bias)

        else:

            if self.bias_counter == 0:
                self.progress_bias.initiate(len(self.bias_files))

            fits_data, fits_header = self.get_fits_data_and_header(self.bias_files[self.bias_counter])
            self.bias_frames.append(fits_data)

            try:
                bias_exptime = float(fits_header[self.log.get_param('exposure_time_key')])
            except:
                bias_exptime = 0.0

            self.bias_exp_time.append(bias_exptime)

            print('{0}: median = {1}, exp.time = {2}'.format(
                self.bias_files[self.bias_counter], np.nanmedian(self.bias_frames[-1]), bias_exptime)
            )

            self.progress_bias.update()
            self.bias_counter += 1

            if self.bias_counter >= len(self.bias_files):
                self.progress_bias.show_message('Calculating master bias...')
                self.after(self.get_master_bias)
            else:
                self.after(self.get_bias)

    def get_master_bias(self):

        if self.exit:
            self.after(self.get_dark)

        else:

            if len(self.bias_frames) > 0:

                consistent_exp_time = self.bias_exp_time == np.median(self.bias_exp_time)

                self.bias_frames = [self.bias_frames[ff]
                                    for ff in range(len(self.bias_frames)) if consistent_exp_time[ff]]

                self.master_bias = combine_frames(self.bias_frames,
                                                  method=self.log.get_param('master_bias_method'))

                self.bias_exp_time = np.median(self.bias_exp_time)

            else:
                self.master_bias = 0.0
                self.bias_exp_time = 0.0

            print('Median Bias: ', round(np.nanmedian(self.master_bias), 3))
            print('Bias exp. time: ', self.bias_exp_time)
            self.progress_bias.show_message('Calculating master bias... Completed!')

            self.after(self.get_dark)

    def get_dark(self):

        if self.exit or len(self.dark_files) == 0:
            self.after(self.get_master_dark)

        else:

            if self.dark_counter == 0:
                self.progress_dark.initiate(len(self.dark_files))

            fits_data, fits_header = self.get_fits_data_and_header(self.dark_files[self.dark_counter])

            self.dark_frames.append(
                reduce_dark_frame(fits_data, self.master_bias,
                                  float(fits_header[self.log.get_param('exposure_time_key')]),
                                  self.bias_exp_time)
            )

            print('{0}: median = {1}'.format(
                self.dark_files[self.dark_counter], np.nanmedian(self.dark_frames[-1]))
            )

            self.progress_dark.update()
            self.dark_counter += 1

            if self.dark_counter >= len(self.dark_files):
                self.progress_dark.show_message('Calculating master dark...')
                self.after(self.get_master_dark)
            else:
                self.after(self.get_dark)

    def get_master_dark(self):

        if self.exit:
            self.after(self.get_flat)
        else:

            if len(self.dark_frames) > 0:
                self.master_dark = combine_frames(self.dark_frames,
                                                  method=self.log.get_param('master_dark_method'))

            else:
                self.master_dark = 0.0

            print('Median Dark: ', round(np.nanmedian(self.master_dark), 3))
            self.progress_dark.show_message('Calculating master dark... Completed!')

            self.after(self.get_darkf)

    def get_darkf(self):

        if self.exit or len(self.darkf_files) == 0:
            self.after(self.get_master_darkf)

        else:

            if self.darkf_counter == 0:
                self.progress_darkf.initiate(len(self.darkf_files))

            fits_data, fits_header = self.get_fits_data_and_header(self.darkf_files[self.darkf_counter])

            self.darkf_frames.append(
                reduce_dark_frame(fits_data, self.master_bias,
                                  float(fits_header[self.log.get_param('exposure_time_key')]),
                                  self.bias_exp_time)
            )

            print('{0}: median = {1}'.format(
                self.darkf_files[self.darkf_counter], np.nanmedian(self.darkf_frames[-1]))
            )

            self.progress_darkf.update()
            self.darkf_counter += 1

            if self.darkf_counter >= len(self.darkf_files):
                self.progress_darkf.show_message('Calculating master dark-flat...')
                self.after(self.get_master_darkf)
            else:
                self.after(self.get_darkf)

    def get_master_darkf(self):

        if self.exit:
            self.after(self.get_flat)
        else:

            if len(self.darkf_frames) > 0:
                self.master_darkf = combine_frames(self.darkf_frames,
                                                   method=self.log.get_param('master_darkf_method'))
            else:
                self.master_darkf = self.master_dark

            print('Median Dark-Flat: ', round(np.nanmedian(self.master_darkf), 3))
            self.progress_darkf.show_message('Calculating master dark-flat... Completed!')

            self.after(self.get_flat)

    def get_flat(self):

        if self.exit or len(self.flat_files) == 0:
            self.after(self.get_master_flat)

        else:

            if self.flat_counter == 0:
                self.progress_flat.initiate(len(self.flat_files))

            fits_data, fits_header = self.get_fits_data_and_header(self.flat_files[self.flat_counter])

            self.flat_frames.append(
                reduce_flat_frame(fits_data, self.master_darkf, self.master_bias,
                                  float(fits_header[self.log.get_param('exposure_time_key')]), self.bias_exp_time)
            )

            print('{0}: median = {1}'.format(
                self.flat_files[self.flat_counter], np.nanmedian(self.flat_frames[-1]))
            )

            self.progress_flat.update()
            self.flat_counter += 1

            if self.flat_counter >= len(self.flat_files):
                self.progress_flat.show_message('Calculating master flat...')
                self.after(self.get_master_flat)
            else:
                self.after(self.get_flat)

    def get_master_flat(self):

        if self.exit:
            self.after(self.reduce_science)

        else:
            self.master_flat = compute_master_flat(self.flat_frames,
                                                   method=self.log.get_param('master_flat_method'),
                                                   colour_camera_mode=self.log.get_param('colour_camera_mode'))

            self.progress_flat.show_message('Calculating master flat... Completed!')
            sys.setrecursionlimit(100 * len(self.science_files))
            self.after(self.reduce_science)

    def reduce_science(self):

        # correct each observation_files file

        if self.exit:
            self.after(self.stack)

        else:

            if self.science_counter == 0:
                self.progress_science.initiate(len(self.science_files))

            science_file = self.science_files[self.science_counter]

            # load fits file

            fits_data, fits_header = self.get_fits_data_and_header(science_file)
            saturation = get_saturation(fits_header, self.log.hops_saturation_key)

            # correct with master bias_files, master dark_files and master flat_files

            data_frame = reduce_science_frame(
                fits_data, saturation, self.master_flat, self.master_dark, self.master_bias,
                float(fits_header[self.log.get_param('exposure_time_key')]), self.bias_exp_time
            )

            # Binning and cropping

            data_frame = bin_and_crop_science_frame(
                data_frame, saturation, self.log.get_param('crop_x1'),
                self.log.get_param('crop_x2'), self.log.get_param('crop_y1'), self.log.get_param('crop_y2'),
                self.log.get_param('crop_edge_pixels'), self.log.get_param('bin_fits'),
            )

            # setting the refresh rate based on the computer's performance

            if self.science_counter == 0:
                t0 = time.time()
                _ = mean_std_from_median_mad(data_frame)
                self.fr_time = int(1000 * (time.time()-t0))

            # calculate statistics

            fits_header = add_image_statistics_to_header(data_frame, fits_header, saturation, self.log)

            try:

                fits_header, observation_time = add_timing_and_coordinates_to_header(fits_header, self.log)

                # write the new fits file
                # important to keep it like this for windows!

                time_in_file = observation_time.utc().isoformat()
                time_in_file = time_in_file.split('.')[0]
                time_in_file = time_in_file.replace('-', '_').replace(':', '_').replace('T', '_')

                new_name = '{0}{1}_{2}'.format(
                    self.log.reduction_prefix, time_in_file, science_file.split(os.sep)[-1]
                )

                if self.compress_reduced:
                    primary = pf.PrimaryHDU()
                    image = pf.CompImageHDU()

                else:
                    image = pf.PrimaryHDU()

                for i in fits_header:
                    image.header.set(i, fits_header[i])

                image.data = np.array(data_frame, dtype=np.int32)
                image.header.set(self.log.align_x0_key, False)
                image.header.set(self.log.align_y0_key, False)
                image.header.set(self.log.align_u0_key, False)
                image.header.set(self.log.align_u0_key, False)

                if self.compress_reduced:
                    save_fits(pf.HDUList([primary, image]), os.path.join(self.log.reduction_directory, new_name))
                else:
                    save_fits(pf.HDUList([image]), os.path.join(self.log.reduction_directory, new_name))

                self.all_frames[new_name] = {
                    self.log.mean_key: fits_header[self.log.mean_key],
                    self.log.std_key: fits_header[self.log.std_key],
                    self.log.psf_key: fits_header[self.log.psf_key],
                    self.log.bpp_key: fits_header[self.log.bpp_key],
                    self.log.time_key: fits_header[self.log.time_key],
                    self.log.airmass_key: fits_header[self.log.airmass_key],
                    self.log.get_param('exposure_time_key'): fits_header[self.log.hops_exposure_key],
                    self.log.skip_key: fits_header[self.log.skip_key],
                    self.log.align_x0_key: False,
                    self.log.align_y0_key: False,
                    self.log.align_u0_key: False,
                }

                self.progress_science.update()
                self.science_counter += 1

                if self.science_counter >= len(self.science_files):

                    save_dict(self.all_frames, self.log.all_frames_path)
                    self.log.set_param('reduction_complete', True)
                    self.log.set_param('reduction_version', self.log.version)
                    self.log.set_param('stacking_complete', False)
                    self.log.save_local_log()

                    self.after(self.stack)
                else:
                    if self.progress_science_loop.get() or self.science_counter == 1:
                        self.progress_figure.load_fits(data_frame, fits_header, new_name)
                        self.progress_figure.draw()

                    if len(self.jobs) > self.jobs_completed + 1:
                        self.fr_time += 10
                    else:
                        self.fr_time -= 2
                        self.fr_time = max(self.fr_time, 2)

                    self.after(self.reduce_science, time=self.fr_time)

            except exoclock.errors.ExoClockInputError:
                print('Bad time data, skipping frame: ', science_file)

                self.science_counter += 1

                if self.science_counter >= len(self.science_files):
                    self.after(self.stack)
                else:
                    self.after(self.reduce_science, time=self.fr_time)



    def stack(self):
        if self.exit:
            self.after(self.proceed)

        else:

            if len(self.all_frames) == 0:
                self.all_frames = open_dict(self.log.all_frames_path)

            exp_key = self.log.get_param('exposure_time_key')
            t_eff_setting = self.log.get_param('effective_exposure')
            if t_eff_setting not in ["OFF", "off", "Off", None, ""]:

                t_eff = float(t_eff_setting)

                print("\n--------------------------------------------------------------------------")
                print("FRAME STACKING")
                print("--------------------------------------------------------------------------")
                print(f"Target effective exposure: {t_eff:.1f} sec")  ##### read the reduced, then sort

                red_dir = self.log.reduction_directory
                old_stacked_files = [f for f in os.listdir(red_dir) if f.startswith('stk_')]
                if len(old_stacked_files) > 0:
                    for f in old_stacked_files:
                        os.remove(os.path.join(red_dir, f))

                print("Loading file data...")
                jds = []
                T_before = 0

                for f in self.all_frames:

                    if self.exit:
                        self.after(self.proceed)

                    else:

                        exp = float(self.all_frames[f][exp_key])
                        T_before += exp

                        jds.append([self.all_frames[f][self.log.time_key], exp, {
                            'file': os.path.join(red_dir, f), 'exp': exp, 'jd': self.all_frames[f][self.log.time_key],
                        }])

                jds = sorted(jds)

                file_metadata = [ff[2] for ff in jds]
                jd = np.array([ff[0] for ff in jds])
                exp = np.array([ff[1] for ff in jds])

                overheads = round(np.median((jd[1:] - jd[:-1]) * 24 * 3600 - exp[:-1]), 2)

                expected_effective_exp_time = t_eff + (round(t_eff / file_metadata[0]['exp'], 0) - 1) * overheads

                print(f"Total exposure before stacking: {T_before:.1f} sec")
                print(f"Number of frames to stack: {len(file_metadata)}")
                print(f"Overheads per frame: {overheads}")
                print(f"Targeted exposure time (including overheads): {expected_effective_exp_time}")

                self.progress_stacking.initiate(len(file_metadata))

                stack_dir_temp = os.path.join(red_dir, "stacked_temp")
                if os.path.exists(stack_dir_temp):
                    shutil.rmtree(stack_dir_temp)
                os.makedirs(stack_dir_temp, exist_ok=True)
                temp_group = []
                temp_group_header = []
                temp_group_metadata = []
                stack_count = 0
                threshold = 5

                new_all_frames = {}

                for idx, meta in enumerate(file_metadata):

                    if self.exit:
                        self.after(self.proceed)

                    else:

                        with open_fits(meta['file']) as hdul:
                            if len(hdul) > 1 and hdul[1].data is not None:
                                data = hdul[1].data
                                header = hdul[1].header
                            elif hdul[0].data is not None:
                                data = hdul[0].data
                                header = hdul[0].header
                            else:
                                print(f"Warning: No valid data found in {meta['file']}")
                                continue

                        if len(temp_group) == 0:

                            temp_group.append(np.array(data, dtype=np.float32))
                            temp_group_header.append(header)
                            temp_group_metadata.append(meta)

                        else:
                            temp_exp = ((temp_group_metadata[-1]['jd'] - temp_group_metadata[0]['jd']) * 24 * 3600
                                        + temp_group_metadata[-1]['exp'])
                            temp_exp_with_current = ((meta['jd'] - temp_group_metadata[0]['jd']) * 24 * 3600
                                                     + meta['exp'])

                            error_without = expected_effective_exp_time - temp_exp
                            error_with = temp_exp_with_current - expected_effective_exp_time

                            if error_without < threshold and error_with > threshold:
                                should_include_current = False
                                true_exp_time = np.sum([ff['exp'] for ff in temp_group_metadata])
                                if true_exp_time < 0.8 * t_eff:
                                    should_stack = False
                                    skip_stack = True
                                    reset_stack_with_current = True
                                else:
                                    should_stack = True
                                    skip_stack = False
                            elif error_without > threshold and error_with < -threshold:
                                should_include_current = True
                                should_stack = False
                                skip_stack = False
                            elif error_without > threshold and abs(error_with) < threshold:
                                should_include_current = True
                                true_exp_time = np.sum([ff['exp'] for ff in temp_group_metadata]) + meta['exp']
                                if true_exp_time < 0.8 * t_eff:
                                    should_stack = False
                                    skip_stack = True
                                    reset_stack_with_current = False
                                else:
                                    should_stack = True
                                    skip_stack = False
                            else:
                                should_include_current = False
                                should_stack = False
                                skip_stack = True
                                reset_stack_with_current = True

                            if should_include_current:

                                temp_group.append(np.array(data, dtype=np.float32))
                                temp_group_header.append(header)
                                temp_group_metadata.append(meta)

                            if should_stack:

                                stacked, new_header, effective_exposure = stack_frames_calc(
                                    temp_group, temp_group_header, self.log, stack_count, t_eff
                                )

                                stacked_file = f"stk_{stack_count + 1:04d}.fits"
                                outname = os.path.join(stack_dir_temp, stacked_file)

                                if self.compress_reduced:
                                    primary = pf.PrimaryHDU()  ############ save frame
                                    image = pf.CompImageHDU(data=stacked.astype(np.int32), header=new_header)
                                    save_fits(pf.HDUList([primary, image]), outname)

                                else:
                                    image = pf.PrimaryHDU(data=stacked.astype(np.int32), header=new_header)
                                    save_fits(pf.HDUList([image]), outname)

                                print(
                                    f"Created stack {stack_count + 1}: {len(temp_group)} frames, "
                                    f"{effective_exposure:.1f}s effective exposure"
                                )

                                new_all_frames[stacked_file] = {
                                    self.log.mean_key: new_header[self.log.mean_key],
                                    self.log.std_key: new_header[self.log.std_key],
                                    self.log.psf_key: new_header[self.log.psf_key],
                                    self.log.bpp_key: new_header[self.log.bpp_key],
                                    self.log.time_key: new_header[self.log.time_key],
                                    self.log.airmass_key: new_header[self.log.airmass_key],
                                    exp_key: new_header[self.log.get_param('exposure_time_key')],
                                    'TRUEEXP': new_header['TRUEEXP'],
                                    self.log.skip_key: new_header[self.log.skip_key],
                                    self.log.align_x0_key: False,
                                    self.log.align_y0_key: False,
                                    self.log.align_u0_key: False,
                                }

                                stack_count += 1

                                temp_group = []
                                temp_group_header = []
                                temp_group_metadata = []

                            if skip_stack:

                                rejected_actual_time = np.sum([ff['exp'] for ff in temp_group_metadata])

                                print(f"\nSkipping {len(temp_group)} frames, with actual exposure time "
                                      f"of {rejected_actual_time} s within time-span of {round(temp_exp, 2)} s. "
                                      f"Adding next frame would give a time-span "
                                      f"of {round(temp_exp_with_current, 2)} s")

                                if reset_stack_with_current:
                                    temp_group = [np.array(data, dtype=np.float32)]
                                    temp_group_header = [header]
                                    temp_group_metadata = [meta]
                                else:
                                    temp_group = []
                                    temp_group_header = []
                                    temp_group_metadata = []


                        self.progress_stacking.update()
                        self.progress_stacking.show_message(
                            f"Processing frame {idx + 1}/{len(file_metadata)} - Created {stack_count} stacks")
                        self.update()

                if self.exit:
                    self.after(self.proceed)

                else:

                    stacked_files = [f for f in os.listdir(stack_dir_temp)]

                    print("Created {stack_count} stacked frames in sec")
                    print("\nAdding stacked frames to reduced data directory...")
                    self.progress_stacking.show_message(f"Stacking complete - Created {stack_count} stacks")
                    for f in stacked_files:
                        src = os.path.join(stack_dir_temp, f)
                        dst = os.path.join(red_dir, f)
                        shutil.move(src, dst)

                    shutil.rmtree(stack_dir_temp)
                    print(f"Added {len(stacked_files)} stacked frames to reduced data directory\n")


                    if len(new_all_frames) > 0:
                        self.all_frames = new_all_frames
                        save_dict(self.all_frames, self.log.all_frames_path)
                        self.log.set_param('using_stacked_frames', True)
                        self.log.set_param('alignment_complete', False)
                        self.log.set_param('photometry_complete', False)
                        self.log.set_param('fitting_complete', False)
                        deleted_count = 0
                        for f in os.listdir(red_dir):
                            if not f.startswith('stk_'):
                                try:
                                    os.remove(os.path.join(red_dir, f))
                                    deleted_count += 1
                                except Exception as e:
                                    print(f"Warning: Could not delete original reduced frame {f}: {e}")
                        print(f"Deleted {deleted_count} original reduced frames after successful stacking")
                        self.log.set_param('stacking_complete', True)
                        self.log.save_local_log()
                        print(f"Updated all_frames.pickle with {len(new_all_frames)} stacked frames")
                    else:
                        print("ERROR: Could not update all_frames.pickle")
                    print("\nSTACKING FINISHED")


                    stacked_files = [f for f in os.listdir(self.log.reduction_directory)
                                     if f.startswith('stk_')]
                    self.showinfo('Stacking Complete',
                                  f'Reduction and stacking completed successfully!\n\n'
                                  f'Created {len(stacked_files)} stacked frames.')

                    self.after(self.proceed)

            else:
                self.log.set_param('stacking_complete', True)
                self.log.save_local_log()

                self.after(self.proceed)


    def proceed(self):

        if self.exit:
            self.close()
        else:
            self.log.set_param('proceed', True)
            self.close()



