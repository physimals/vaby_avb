# Example fitting of ASL model
#
# This example uses the sample multi-PLD data from the FSL course
import sys

import matplotlib.pyplot as plt
import nibabel as nib

from avb import run

ASLDATA = "asldata_diff.nii.gz"
MASK = "asldata_mask.nii.gz"

model = "aslrest"
outdir = "asl_example_out"

options = {
    "tau" : 1.8,
    "casl" : True,
    "plds" : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "repeats" : [8],
    "slicedt" : 0.0452,
    "save_mean" : True,
    "save_noise" : True,
    "save_runtime" : True,
    "save_free_energy" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "save_input_data" : True,
    "save_var" : True,
    "save_post" : True,
    "log_stream" : sys.stdout,
    "max_iterations" : 20,
    "param_overrides" : {
#        "ftiss" : {"prior_type" : "N", "dist" : "Normal"},
#        "fblood" : {"prior_type" : "A", "dist" : "ClippedNormal"},
    },
}

if "--inferart" in sys.argv:
    options["inferart"] = True
    outdir += "_art"
if "--disp" in sys.argv:
    model = "aslrest_disp"
    options["disp"] = "gamma"
    outdir += "_gammadisp"
if "--adam" in sys.argv:
    options["use_adam"] = True
    options["max_iterations"] = 100
    outdir += "_adam"

if "--run-fabber" in sys.argv:
    # Run fabber as a comparison
    fabber_cmd = "fabber_asl --model=aslrest --method=spatialvb --noise=white --data=asldata_diff --mask=asldata_mask --max-iterations=20 --tau=1.8 --casl --batsd=1.0 --bat=1.3 --ti1=2.05 --ti2=2.3 --ti3=2.55 --ti4=2.8 --ti5=3.05 --ti6=3.3 --slicedt=0.0454 --inctiss --infertiss --incbat --inferbat --repeats=8 --overwrite --save-model-fit "
    import os
    # Without arterial component
    os.system(fabber_cmd + "--output=asl_example_fabber_out")
    # With arterial component
    os.system(fabber_cmd + "--incart --inferart --output=asl_example_fabber_out_art")
    # Spatial VM without arterial component
    os.system(fabber_cmd + "--param-spatial-priors=MN+ --output=asl_example_fabber_out_svb")

if "--avb-then-adam" in sys.argv:
    runtime, avb = run(ASLDATA, model, outdir + "_init", mask=MASK, **options)
    options["initial_posterior"] = "%s_init/posterior.nii.gz" % outdir
    options["use_adam"] = True
    options["max_iterations"] = 500
    options["learning_rate"] = 0.2
    runtime, avb = run(ASLDATA, model, outdir, mask=MASK, **options)
else:
    runtime, avb = run(ASLDATA, model, outdir, mask=MASK, **options)

if "--display" in sys.argv:
    # Display a single slice (z=10)
    ftiss_img = nib.load("%s/mean_ftiss.nii.gz" % outdir).get_fdata()
    delttiss_img = nib.load("%s/mean_delttiss.nii.gz" % outdir).get_fdata()
    plt.figure("F")
    plt.imshow(ftiss_img[:, :, 10].squeeze())
    plt.figure("delt")
    plt.imshow(delttiss_img[:, :, 10].squeeze())
    plt.show()
