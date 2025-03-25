import logging
import sys
import argparse

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-g", "--logpath", type=ascii, dest="log_path", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("-s", "--scale", type=float, default=1.,
                        help="scale geometry by int factor")
    parser.add_argument("--esdg", action="store_true", default=False,
                        help="enable entropy-stable for inviscid terms. [OFF]")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--disable-fallbacks", action="store_true", default=False,
                        help="prevent lazy compile from using slow fallbacks.")
    parser.add_argument("--nolog", action="store_true", default=False,
                        help="enable sql logging with logpyle [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")
    parser.add_argument("--numpy", action="store_true",
        help="use numpy-based eager actx.")

    args = parser.parse_args()

    # for writing output
    casename = "prediction"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")

    from mirgecom.simutil import ApplicationOptionsError
    if args.esdg and not (args.lazy or args.numpy):
        raise ApplicationOptionsError("ESDG requires lazy or numpy context.")

    from mirgecom.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(
        lazy=args.lazy, distributed=True, profiling=args.profile, numpy=args.numpy)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    log_path = "log_data"
    if args.log_path:
        log_path = args.log_path.replace("'", "")

    print(f"Running {sys.argv[0]}\n")

    from y3prediction.prediction import main
    main(actx_class, restart_filename=restart_filename,
         target_filename=target_filename,
         user_input_file=input_file, log_path=log_path,
         disable_logpyle=args.nolog, geom_scale=args.scale,
         use_overintegration=args.overintegration or args.esdg,
         casename=casename, use_esdg=args.esdg,
         disable_fallbacks=args.disable_fallbacks)
