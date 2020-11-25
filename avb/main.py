"""
Implementation of command line tool for AVB

Examples::

    avb --data=asldata.nii.gz --mask=bet_mask.nii.gz
        --model=aslrest --epochs=200 --output=avb_out
"""
import os
import sys
import logging
import logging.config
import argparse
import re

import nibabel as nib
import numpy as np

from . import __version__, Avb
from svb.utils import ValueList
from svb import get_model_class
from svb.data import VolumetricModel, SurfaceModel

USAGE = "avb <options>"

class AvbArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser for AVB options
    """

    PARAM_OPTIONS = {
        "dist" : str,
        "prior_mean" : float,
        "prior_var" : float,
        "prior_type" : str,
        "post_mean" : float,
        "post_var" : float,
    }

    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="avb", usage=USAGE, add_help=False, **kwargs)

        group = self.add_argument_group("Main Options")
        group.add_argument("--data",
                         help="Timeseries input data")
        group.add_argument("--mask",
                         help="Optional voxel mask")
        group.add_argument("--post-init", dest="post_init_fname",
                         help="Initialize posterior from data file saved using --output-post")
        group.add_argument("--model", dest="model_name",
                         help="Model name")
        group.add_argument("--output",
                         help="Output folder name",
                         default="avb_out")
        group.add_argument("--log-level",
                         help="Logging level - defaults to INFO")
        group.add_argument("--log-config",
                         help="Optional logging configuration file, overrides --log-level")
        group.add_argument("--help", action="store_true", default=False,
                         help="Display help")
        
        group = self.add_argument_group("Inference options")
        group.add_argument("--max-iterations",
                         help="Max number of iterations",
                         type=int, default=20)
        group.add_argument("--use-adam", action="store_true", default=False,
                         help="Directly maximise free energy using Adam optimizer")
        group.add_argument("--learning-rate",
                         help="Learning rate for Adam optimizer",
                         type=float, default=0.5)
        group.add_argument("--init-leastsq",
                         help="Do an initial least-squares fit before optimizing full free energy cost",
                         action="store_true", default=False)

        group = self.add_argument_group("Output options")
        group.add_argument("--save-var",
                         help="Save parameter variance",
                         action="store_true", default=False)
        group.add_argument("--save-std",
                         help="Save parameter standard deviation",
                         action="store_true", default=False)
        group.add_argument("--save-param-history",
                         help="Save parameter history by epoch",
                         action="store_true", default=False)
        group.add_argument("--save-noise",
                         help="Save noise parameter",
                         action="store_true", default=False)
        group.add_argument("--save-free-energy",
                         help="Save free energy",
                         action="store_true", default=False)
        group.add_argument("--save-free-energy-history",
                         help="Save free energy history by iteration",
                         action="store_true", default=False)
        group.add_argument("--save-model-fit",
                         help="Save model fit",
                         action="store_true", default=False)
        group.add_argument("--save-post", "--save-posterior",
                         help="Save full posterior distribution",
                         action="store_true", default=False)

    def parse_args(self, argv=None, namespace=None):
        # Parse built-in fixed options but skip unrecognized options as they may be
        #  model-specific option or parameter-specific optionss.
        options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        # Now we should know the model, so we can add it's options and parse again
        if options.model_name:
            group = self.add_argument_group("%s model options" % options.model_name.upper())
            for model_option in get_model_class(options.model_name).OPTIONS:
                kwargs = {
                    "help" : model_option.desc,
                    "type" : model_option.type,
                    "default" : model_option.default,
                }
                if model_option.units:
                    kwargs["help"] += " (%s)" % model_option.units
                if model_option.default is not None:
                    kwargs["help"] += " - default %s" % str(model_option.default)
                else:
                    kwargs["help"] += " - no default"

                if model_option.type == bool:
                    kwargs["action"] = "store_true"
                    kwargs.pop("type")
                group.add_argument(*model_option.clargs, **kwargs)
            options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        if options.help:
            self.print_help()
            sys.exit(0)

        # Support arguments of the form --param-<param name>-<param option>
        # (e.g. --param-ftiss-mean=4.4 --param-delttiss-prior-type M)
        param_arg = re.compile("--param-(\w+)-([\w-]+)")
        options.param_overrides = {}
        consume_next_arg = None
        for arg in extras:
            if consume_next_arg:
                if arg.startswith("-"):
                    raise ValueError("Value for parameter option cannot start with - : %s" % arg)
                param, thing = consume_next_arg
                options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](arg)
                consume_next_arg = None
            else:
                kv = arg.split("=", 1)
                key = kv[0]
                match = param_arg.match(key)
                if match:
                    param, thing = match.group(1), match.group(2)

                    # Use underscore for compatibility with kwargs
                    thing = thing.replace("-", "_")
                    if thing not in self.PARAM_OPTIONS:
                        raise ValueError("Unrecognized parameter option: %s" % thing)

                    if param not in options.param_overrides:
                        options.param_overrides[param] = {}
                    if len(kv) == 2:
                        options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](kv[1])
                    else:
                        consume_next_arg = (param, thing)
                else:
                    raise ValueError("Unrecognized argument: %s" % arg)

        return options

def main():
    """
    Command line tool entry point
    """
    try:
        arg_parser = AvbArgumentParser()
        options = arg_parser.parse_args()

        if not options.data:
            raise ValueError("Input data not specified")
        if not options.model_name:
            raise ValueError("Model name not specified")

        # Fixed for CL tool
        options.save_mean = True
        options.save_runtime = True
        options.save_log = True

        welcome = "Welcome to AVB %s" % __version__
        print(welcome)
        print("=" * len(welcome))
        runtime, _ = run(log_stream=sys.stdout, **vars(options))
        print("FINISHED - runtime %.3fs" % runtime)
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write("ERROR: %s\n" % str(exc))
        import traceback
        traceback.print_exc()

def run(data, model_name, output, mask=None, surfaces=None, **kwargs):
    """
    Run model fitting on a data set

    :param data: File name of 4D NIFTI data set containing data to be fitted
    :param model_name: Name of model we are fitting to
    :param output: output directory, will be created if it does not exist
    :param mask: Optional file name of 3D Nifti data set containing data voxel mask

    All keyword arguments are passed to constructor of the model, the ``Avb``
    object and the ``Avb.run`` method.
    """
    # Create output directory
    _makedirs(output, exist_ok=True)
    
    setup_logging(output, **kwargs)
    log = logging.getLogger(__name__)
    log.info("AVB %s", __version__)

    # Initialize the data model which contains data dimensions, number of time
    # points, list of unmasked voxels, etc
    if surfaces is None: 
        data_model = VolumetricModel(data, mask, **kwargs)
    else:
        data_model = SurfaceModel(data, surfaces, mask, **kwargs)
    
    # Create the generative model
    fwd_model = get_model_class(model_name)(data_model, **kwargs)
    fwd_model.log_config()

    # Get the time points from the model in node space
    tpts = fwd_model.tpts()

    history = kwargs.get("save_free_energy_history", False) or kwargs.get("save_param_history", False)
    avb = Avb(tpts, data_model, fwd_model, **kwargs)
    runtime, _ret = _runtime(avb.run, record_history=history, **kwargs)
    log.info("DONE: %.3fs", runtime)

    _makedirs(output, exist_ok=True)
    params = [p.name for p in fwd_model.params]
    
    # Write out parameter mean and variance images
    mean = avb.model_mean
    variances = avb.model_var
    for idx, param in enumerate(params):
        if kwargs.get("save_mean", False):
            data_model.nifti_image(mean[idx]).to_filename(os.path.join(output, "mean_%s.nii.gz" % param))
        if kwargs.get("save_var", False):
            data_model.nifti_image(variances[idx]).to_filename(os.path.join(output, "var_%s.nii.gz" % param))
        if kwargs.get("save_std", False):
            data_model.nifti_image(np.sqrt(variances[idx])).to_filename(os.path.join(output, "std_%s.nii.gz" % param))

    if kwargs.get("save_noise", False):
        if kwargs.get("save_mean", False):
            data_model.nifti_image(avb.noise_mean).to_filename(os.path.join(output, "mean_noise.nii.gz"))
        if kwargs.get("save_var", False):
            data_model.nifti_image(avb.noise_var).to_filename(os.path.join(output, "var_noise.nii.gz"))
        if kwargs.get("save_std", False):
            data_model.nifti_image(np.sqrt(avb.noise_var)).to_filename(os.path.join(output, "std_noise.nii.gz"))

    # Write out voxelwise free energy (and history if required)
    if kwargs.get("save_free_energy", False):
        data_model.nifti_image(avb.free_energy_vox).to_filename(os.path.join(output, "free_energy.nii.gz"))
    if kwargs.get("save_free_energy_history", False):
        data_model.nifti_image(avb.history["free_energy_vox"]).to_filename(os.path.join(output, "free_energy_history.nii.gz"))

    # Write out voxelwise parameter history
    if kwargs.get("save_param_history", False):
        for idx, param in enumerate(params):
            data_model.nifti_image(avb.history["model_mean"][idx]).to_filename(os.path.join(output, "mean_%s_history.nii.gz" % param))

    # Write out modelfit
    if kwargs.get("save_model_fit", False):
        data_model.nifti_image(avb.modelfit).to_filename(os.path.join(output, "modelfit.nii.gz"))

    # Write out posterior
    if kwargs.get("save_post", False):
        post_data = data_model.posterior_data(avb.all_mean, avb.all_cov)
        log.info("Posterior data shape: %s", post_data.shape)
        data_model.nifti_image(post_data).to_filename(os.path.join(output, "posterior.nii.gz"))

    # Write out runtime
    if kwargs.get("save_runtime", False):
        with open(os.path.join(output, "runtime"), "w") as runtime_f:
            runtime_f.write("%f\n" % runtime)

    # Write out input data
    if kwargs.get("save_input_data", False):
        data_model.nifti_image(data_model.data_flattened).to_filename(os.path.join(output, "input_data.nii.gz"))

    log.info("Output written to: %s", output)
    return runtime, avb

def setup_logging(outdir=".", **kwargs):
    """
    Set the log level, formatters and output streams for the logging output

    By default this goes to <outdir>/logfile at level INFO
    """
    # First we clear all loggers from previous runs
    for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ['']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []

    if kwargs.get("log_config", None):
        # User can supply a logging config file which overrides everything else
        logging.config.fileConfig(kwargs["log_config"])
    else:
        # Set log level on the root logger to allow for the possibility of 
        # debug logging on individual loggers
        level = kwargs.get("log_level", "info")
        if not level:
            level = "info"
        level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(level)

        if kwargs.get("save_log", False):
            # Send the log to an output logfile
            logfile = os.path.join(outdir, "logfile")
            logging.basicConfig(filename=logfile, filemode="w", level=level)

        if kwargs.get("log_stream", None) is not None:
            # Can also supply a stream to send log output to as well (e.g. sys.stdout)
            extra_handler = logging.StreamHandler(kwargs["log_stream"])
            extra_handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
            logging.getLogger().addHandler(extra_handler)

def _runtime(runnable, *args, **kwargs):
    """
    Record how long it took to run something
    """
    import time
    start_time = time.time()
    ret = runnable(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time), ret

def _makedirs(data_vol, exist_ok=False):
    """
    Make directories, optionally ignoring them if they already exist
    """
    try:
        os.makedirs(data_vol)
    except OSError as exc:
        import errno
        if not exist_ok or exc.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    main()
