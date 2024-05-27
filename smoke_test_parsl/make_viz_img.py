#!/usr/bin/env python3

import parsl
from parsl.config import Config
from parsl.providers import LSFProvider, SlurmProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import JsrunLauncher, SrunLauncher
from parsl.addresses import address_by_hostname
from parsl.app.app import bash_app, python_app

executor_name = "my_executor"
host = address_by_hostname()
if 'lassen' in host.lower():
    executor = HighThroughputExecutor(
        label=executor_name,
        address=address_by_hostname(),
        worker_debug=True,
        provider=LSFProvider(launcher=JsrunLauncher(overrides='-g 1 -a 1'),
            walltime='02:00:00',
            nodes_per_block=1,
            init_blocks=1,
            max_blocks=1,
            bsub_redirection=True,
            queue='pdebug',
            worker_init=(
                'export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"\n'
                'export PYOPENCL_CTX="port:tesla"\n'
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
            worker_init=(
                "module load spectrum-mpi\n"
                "source emirge/miniforge3/bin/activate mirgeDriver.Y3prediction\n"
                "export XDG_CACHE_HOME=/tmp/$USER/xdg-scratch\n"),
            )
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
    # we need to know a priori what the outfiles files generated will be,
    # so parsl can generate futures for data dependency
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
        parts = file.split('-')
        # Get the third part which contains the dump index
        numbers_part = parts[3]
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
    #print(run_mirge)
    #print(run_mirge.outputs)
    """

    #######
    # 2. monitor the mirgecom restart files to determine when they become available
    #    we do this seperately from the run future, as it won't tell us the files
    #    are complete until the app completes
    #######

    monitor_restart_data = []
    for future in run_mirge.outputs:
        print(future.filename)

        # run mirge and make the viz files
        monitor_restart_data.append(
            monitor_restart(
                file=os.path.basename(future.filename),
                start_time=start_time,
                #outputs=[File(f"{future.filename}.exists")]
                outputs=[File(future.filename)]
            )
        )

    print(monitor_restart_data)

    #######
    # 3. run mirgecom and generate viz_data from restart data
    #######
    make_viz_data = []
    for app_future in monitor_restart_data:
        # first figure out the names of the viz files to be created
        # so we can make data futures for them
        #print(future.filename)
        #print(app_future)
        #print(app_future.outputs)
        future = app_future.outputs[0]
        #print(future.filename)
        restart_name = os.path.basename(future.filename)
        # remove extension
        restart_name = os.path.splitext(restart_name)[0]
        # remove rank number
        restart_name = restart_name[:len(restart_name)-5]
        # get dump number and the casename
        dump_number = restart_name[len(restart_name)-9:]
        case_name = restart_name[:len(restart_name)-10]
        #print(restart_name)
        #print(case_name)
        #print(dump_number)
        # construct dump name
        viz_name_fluid = (f"viz_data/{case_name}-fluid-{dump_number}.pvtu")
        viz_name_wall = (f"viz_data/{case_name}-wall-{dump_number}.pvtu")
        #print(viz_name_fluid)
        #print(viz_name_wall)

        parsl_viz_outfile = []
        parsl_viz_outfile.append(File(os.path.join(os.getcwd(), viz_name_fluid),))
        parsl_viz_outfile.append(File(os.path.join(os.getcwd(), viz_name_wall),))

        # run mirge and make the viz files
        mirge_viz_cmd = build_execution_string(
            yml="make_viz_params.yaml",
            r=f"restart_data/{restart_name}",
            log=False,
            lazy=False)
        make_viz_data.append(
            execute(execution_string=mirge_viz_cmd,
                    stderr="viz_stderr.txt",
                    stdout="viz_stdout.txt",
                    inputs=[future],
                    outputs=parsl_viz_outfile)
        )

    #make_viz_data[0].result()  # wait until it is done
    #print(make_viz_data[0].outputs)
    print("waiting for viz to finish?")
    print(len(make_viz_data))
    print(make_viz_data)
    loop_ind = 0
    for i in make_viz_data:
        print(f"{loop_ind=}")
        print(i)
        print(i.task_status())
        i.result()
        print("after result")
        print(i.outputs)
        print("after outputs")
        loop_ind = loop_ind + 1

    #viz_output = [i.result() for i in make_viz_data]
    #print(viz_output)
    """

    #run_mirge.result()  # wait until mirge execution is done

if __name__ == "__main__":
    main()
