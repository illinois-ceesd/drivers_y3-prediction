#!/usr/bin/env python3

import os
import parsl
from parsl.config import Config
from parsl.providers import LSFProvider, SlurmProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import JsrunLauncher, SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.app.app import bash_app, python_app

executor_name = "my_executor"
import socket
host = socket.gethostname()
if 'lassen' in host.lower():
    executor = HighThroughputExecutor(
        label=executor_name,
        address=address_by_hostname(),
        worker_debug=True,
        cores_per_worker=1,
        max_workers_per_node=44,
        provider=LSFProvider(launcher=JsrunLauncher(),
            walltime='01:00:00',
            nodes_per_block=1,
            init_blocks=1,
            max_blocks=1,
            bsub_redirection=True,
            queue='pdebug',
            worker_init=(
                "export pvpath=/usr/tce/packages/paraview/paraview-gapps/v5.11.0/bin"
                        ),
            project='uiuc')
        )
elif 'quartz' in host.lower():
    executor = HighThroughputExecutor(label=executor_name,
        address=address_by_hostname(),
        worker_debug=True,
        provider=SlurmProvider(launcher=SrunLauncher(overrides=''),
            walltime='01:00:00',
            nodes_per_block=1,
            init_blocks=1,
            max_blocks=1,
            scheduler_options='#SBATCH -q pdebug',
            worker_init=(),
        ),
    )
else:
    #executor = ThreadPoolExecutor(label=executor_name, max_threads=5)
    from parsl.channels import LocalChannel
    from parsl.providers import LocalProvider
    executor = HighThroughputExecutor(
        label=executor_name,
        worker_debug=True,
        cores_per_worker=1,
        provider=LocalProvider(
            channel=LocalChannel(),
            worker_init=(
                "export pvpath=/Applications/ParaView-5.11.0.app/Contents/bin"
                        ),
            init_blocks=1,
            max_blocks=1,
        ),
    )

# used for low priority tasks, not parallel
local_executor_name = "local_executor"
local_executor = ThreadPoolExecutor(label=local_executor_name, max_threads=5)
config = Config(executors=[executor, local_executor], strategy=None)
parsl.load(config)


@bash_app(executors=[executor_name])
def execute(execution_string="",
            stdout="stdout.txt", stderr="stderr.txt",
            inputs=[], outputs=[]):
    return(execution_string)


def build_paraview_execution_string(fluid_file, dump, path,
                                    wall_file=None,
                                    config=None,
                                    ):
    """ Generate and return the driver command line based on inputs

        Parameters
        ----------
        fluid_file: the fluid .pvtu file to pass to the -f argument of the viz driver
        wall_file: the wall .pvtu file to pass to the -w argument of the viz driver
        config: python configuration file for setting paraview viz parameters
        path: the path to where the results will be put
        dump: the dump index for numbering
    """

    pvpath = "/Applications/ParaView-5.11.0.app/Contents/bin"
    # can't have mpirun here, need to let the executor set whatever that should be
    pvpath = os.environ.get('pvpath')
    #print(f"{pvpath=}")
    cmd = f"{pvpath}/pvpython paraview-driver.py -f {fluid_file}"
    cmd += f" -p {path}"
    cmd += f" -d {dump}"
    if wall_file:
        cmd += f" -w {wall_file}"
    if config:
        cmd += f" -i {config}"

    print(f"\n\n{cmd}\n\n")
    return cmd


def main():

    #######
    # 1. run paraview post processing
    #######

    # get a list of the output files
    # this just runs viz on every *.pvtu in viz_data
    import os
    import re
    path = os.getcwd()
    viz_dir = os.path.join(path,"viz_data")
    all_files = os.listdir(viz_dir)
    viz_files = []
    for file in all_files:
        # only get the .pvtu files
        ext = os.path.splitext(file)[1]
        if ext == ".pvtu":
            # only consider the fluid files for now
            if re.search(r"fluid", file):
                file_path = os.path.join(viz_dir, file)
                viz_files.append(file_path)

    print("Running paraview on the following files:")
    print(*viz_files, sep="\n")
    # check for viz directories, this keeps the paraview view driver from making them
    # avoiding possible race conditions
    slice_img_dir = viz_dir + "/slice_img"
    if not os.path.exists(slice_img_dir):
        os.makedirs(slice_img_dir)

    from parsl.data_provider.files import File
    run_paraview = []
    for file in viz_files:
        viz_data = File(os.path.join(os.getcwd(), file),)
        # Split the filename by '-'
        parts = os.path.basename(file).split('-')
        # Get the third part which contains the dump index
        numbers_part = parts[2]
        # Remove the extension
        numbers_part = numbers_part.split('.')[0]

        paraview_cmd = build_paraview_execution_string(
            config="viz_config.py", fluid_file=file, wall_file=None,
            dump=numbers_part, path=viz_dir)
        run_paraview.append(
            execute(execution_string=paraview_cmd,
                    stderr="paraview_stderr.txt",
                    stdout="paraview_stdout.txt"))

    #run_paraview.result()  # wait until it is done
    outputs = [r.result() for r in run_paraview]

if __name__ == "__main__":
    main()
