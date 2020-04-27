var debayer_files_list = [ // enabled, image
   {debayer_files_list}
];

var debayer_output_folder = {debayer_output_folder}

var registration_files_list = [
   {registration_files_list}
];

var registration_reference_view = {registration_reference_file}

var registration_output_folder = {registration_output_folder}



var integration_files_list = [ // enabled, path, drizzlePath, localNormalizationDataPath
   {integration_files_list}
];




var P = new Debayer;
P.cfaPattern = Debayer.prototype.RGGB;
P.debayerMethod = Debayer.prototype.VNG;
P.fbddNoiseReduction = 0;
P.evaluateNoise = false;
P.noiseEvaluationAlgorithm = Debayer.prototype.NoiseEvaluation_MRS;
P.showImages = true;
P.cfaSourceFilePath = "";
P.targetItems = debayer_files_list
P.noGUIMessages = false;
P.inputHints = "raw cfa";
P.outputHints = "";
P.outputDirectory = debayer_output_folder;
P.outputExtension = ".xisf";
P.outputPrefix = "";
P.outputPostfix = "_d";
P.overwriteExistingFiles = true;
P.onError = Debayer.prototype.OnError_Continue;
P.useFileThreads = true;
P.fileThreadOverload = 1.00;
P.maxFileReadThreads = 1;
P.maxFileWriteThreads = 1;

P.executeGlobal()




var P = new StarAlignment;
P.structureLayers = 5;
P.noiseLayers = 0;
P.hotPixelFilterRadius = 1;
P.noiseReductionFilterRadius = 0;
P.sensitivity = 0.100;
P.peakResponse = 0.80;
P.maxStarDistortion = 0.500;
P.upperLimit = 1.000;
P.invert = false;
P.distortionModel = "";
P.undistortedReference = false;
P.distortionCorrection = false;
P.distortionMaxIterations = 20;
P.distortionTolerance = 0.005;
P.distortionAmplitude = 2;
P.localDistortion = true;
P.localDistortionScale = 256;
P.localDistortionTolerance = 0.050;
P.localDistortionRejection = 2.50;
P.localDistortionRejectionWindow = 64;
P.localDistortionRegularization = 0.010;
P.matcherTolerance = 0.0500;
P.ransacTolerance = 2.00;
P.ransacMaxIterations = 2000;
P.ransacMaximizeInliers = 1.00;
P.ransacMaximizeOverlapping = 1.00;
P.ransacMaximizeRegularity = 1.00;
P.ransacMinimizeError = 1.00;
P.maxStars = 0;
P.fitPSF = StarAlignment.prototype.FitPSF_DistortionOnly;
P.psfTolerance = 0.50;
P.useTriangles = false;
P.polygonSides = 5;
P.descriptorsPerStar = 20;
P.restrictToPreviews = true;
P.intersection = StarAlignment.prototype.MosaicOnly;
P.useBrightnessRelations = false;
P.useScaleDifferences = false;
P.scaleTolerance = 0.100;
P.referenceImage = registration_reference_view
P.referenceIsFile = true;
P.targets = registration_files_list;
P.inputHints = "";
P.outputHints = "";
P.mode = StarAlignment.prototype.RegisterMatch;
P.writeKeywords = true;
P.generateMasks = false;
P.generateDrizzleData = true;
P.generateDistortionMaps = false;
P.frameAdaptation = false;
P.randomizeMosaic = false;
P.noGUIMessages = true;
P.useSurfaceSplines = false;
P.extrapolateLocalDistortion = true;
P.splineSmoothness = 0.050;
P.pixelInterpolation = StarAlignment.prototype.Auto;
P.clampingThreshold = 0.30;
P.outputDirectory = registration_output_folder;
P.outputExtension = ".xisf";
P.outputPrefix = "";
P.outputPostfix = "_r";
P.maskPostfix = "_m";
P.distortionMapPostfix = "_dm";
P.outputSampleFormat = StarAlignment.prototype.SameAsTarget;
P.overwriteExistingFiles = true;
P.onError = StarAlignment.prototype.Continue;
P.useFileThreads = true;
P.fileThreadOverload = 1.20;
P.maxFileReadThreads = 1;
P.maxFileWriteThreads = 1;

P.executeGlobal()








var P = new ImageIntegration;
P.images = integration_files_list;
P.inputHints = "";
P.combination = ImageIntegration.prototype.Average;
P.weightMode = ImageIntegration.prototype.NoiseEvaluation;
P.weightKeyword = "";
P.weightScale = ImageIntegration.prototype.WeightScale_IKSS;
P.ignoreNoiseKeywords = false;
P.normalization = ImageIntegration.prototype.AdditiveWithScaling;
// P.rejection = ImageIntegration.prototype.LinearFit;
P.rejection = ImageIntegration.prototype.SigmaClip;
P.rejectionNormalization = ImageIntegration.prototype.Scale;
P.minMaxLow = 1;
P.minMaxHigh = 1;
P.pcClipLow = 0.200;
P.pcClipHigh = 0.100;
P.sigmaLow = 4.000;
P.sigmaHigh = 3.000;
P.winsorizationCutoff = 5.000;
P.linearFitLow = 5.000;
P.linearFitHigh = 4.000;
P.esdOutliersFraction = 0.30;
P.esdAlpha = 0.05;
P.esdLowRelaxation = 1.50;
P.ccdGain = 1.00;
P.ccdReadNoise = 10.00;
P.ccdScaleNoise = 0.00;
P.clipLow = true;
P.clipHigh = true;
P.rangeClipLow = true;
P.rangeLow = 0.000000;
P.rangeClipHigh = false;
P.rangeHigh = 0.980000;
P.mapRangeRejection = true;
P.reportRangeRejection = false;
P.largeScaleClipLow = false;
P.largeScaleClipLowProtectedLayers = 2;
P.largeScaleClipLowGrowth = 2;
P.largeScaleClipHigh = false;
P.largeScaleClipHighProtectedLayers = 2;
P.largeScaleClipHighGrowth = 2;
P.generate64BitResult = false;
P.generateRejectionMaps = true;
P.generateIntegratedImage = true;
P.generateDrizzleData = false;
P.closePreviousImages = false;
P.bufferSizeMB = 16;
P.stackSizeMB = 1024;
P.autoMemorySize = true;
P.autoMemoryLimit = 0.75;
P.useROI = false;
P.roiX0 = 0;
P.roiY0 = 0;
P.roiX1 = 0;
P.roiY1 = 0;
P.useCache = true;
P.evaluateNoise = true;
P.mrsMinDataFraction = 0.010;
P.subtractPedestals = true;
P.truncateOnOutOfRange = false;
P.noGUIMessages = true;
P.useFileThreads = true;
P.fileThreadOverload = 1.00;

P.executeGlobal()