import os

def create_pixinsight_preprocess_script(output_folder, calibrated_folder, image_filenames):
	full_calibrated_image_filenames = list(map(lambda s: os.path.join(calibrated_folder, s), image_filenames))

	# calibrated_folder = os.path.join(output_folder, 'calibrated')
	# debayered_folder = os.path.join(output_folder, 'debayered')
	# registered_folder = os.path.join(output_folder, 'registered')
	calibrated_folder = output_folder + '/calibrated'
	debayered_folder = output_folder + '/debayered'
	registered_folder = output_folder + '/registered'

	if not os.path.exists(debayered_folder): os.mkdir(debayered_folder)
	if not os.path.exists(registered_folder): os.mkdir(registered_folder)

	base_filenames = list(map(lambda s: s.split('.')[0], image_filenames))

	# calibrated_filenames = list(map(lambda s: os.path.join(calibrated_folder, s + '.tif'), base_filenames))
	# debayered_filenames = list(map(lambda s: os.path.join(debayered_folder, s + '_d.xisf'), base_filenames))
	# registered_filenames = list(map(lambda s: os.path.join(registered_folder, s + '_d_r.xisf'), base_filenames))
	calibrated_filenames = list(map(lambda s: calibrated_folder + '/' +  s + '.tif', base_filenames))
	debayered_filenames = list(map(lambda s: debayered_folder + '/' + s + '_d.xisf', base_filenames))
	registered_filenames = list(map(lambda s: registered_folder + '/' +  s + '_d_r.xisf', base_filenames))

	registration_reference_file = debayered_filenames[len(debayered_filenames) // 2]

	debayer_files_list_str = '\n'.join(['[true, "%s"],' % fn for fn in calibrated_filenames])
	debayer_output_folder_str = '"%s"' % debayered_folder
	registration_files_list_str = '\n'.join(['[true, true, "%s"],' % fn for fn in debayered_filenames])
	registration_refernce_file_str = '"%s"' % registration_reference_file
	registration_output_folder_str = '"%s"' % registered_folder
	integration_files_list_str = '\n'.join(['[true, "%s", "", ""],' % fn for fn in registered_filenames])

	pi_template = open('pixinsight_debayer_register_stack_template.js').read()

	pi_customized = pi_template.format(debayer_files_list = debayer_files_list_str,
										debayer_output_folder = debayer_output_folder_str,
										registration_files_list = registration_files_list_str,
										registration_reference_file = registration_refernce_file_str,
										registration_output_folder = registration_output_folder_str,
										integration_files_list = integration_files_list_str
										)

	script_filename = os.path.join(output_folder, 'pi_script.js')
	open(script_filename, 'w').write(pi_customized)

	return script_filename

def test():
	top_folder = 'K:/pi_preprocess_testing'

	calibrated_folder = os.path.join(top_folder, 'calibrated')
	image_filenames = list(os.listdir(calibrated_folder))

	create_pixinsight_preprocess_script(top_folder, calibrated_folder, image_filenames)

if __name__ == "__main__":
	test()