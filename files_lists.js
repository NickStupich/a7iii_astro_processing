var debayer_files_list = [ // enabled, image
   [true, "K:/scripting_test/calibrated/DSC03631.tif"],
   [true, "K:/scripting_test/calibrated/DSC03632.tif"],
   [true, "K:/scripting_test/calibrated/DSC03633.tif"],
   [true, "K:/scripting_test/calibrated/DSC03634.tif"]
];

var debayer_output_folder = "K:/scripting_test/debayered"

var registration_files_list = [
   [true, true, "K:/scripting_test/debayered/DSC03631_d.xisf"],
   [true, true, "K:/scripting_test/debayered/DSC03632_d.xisf"],
   [true, true, "K:/scripting_test/debayered/DSC03633_d.xisf"],
   [true, true, "K:/scripting_test/debayered/DSC03634_d.xisf"]
];

var registration_reference_view = "K:/scripting_test/debayered/DSC03631_d.xisf"

var registration_output_folder = "K:/scripting_test/registered"



var integration_files_list = [ // enabled, path, drizzlePath, localNormalizationDataPath
   [true, "K:/scripting_test/registered/DSC03631_d_r.xisf", "", ""],
   [true, "K:/scripting_test/registered/DSC03632_d_r.xisf", "", ""],
   [true, "K:/scripting_test/registered/DSC03633_d_r.xisf", "", ""],
   [true, "K:/scripting_test/registered/DSC03634_d_r.xisf", "", ""]
];